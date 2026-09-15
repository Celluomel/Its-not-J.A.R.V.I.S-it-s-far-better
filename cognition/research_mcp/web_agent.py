"""
WebAgent v23 — unified ranking (RSS + providers)
=================================================

NEW IN v23
-----------
• Global ranking across ALL sources (RSS, Brave, Claude, SerpAPI)
• Unified scoring: relevance + recency + credibility + source_weight
• RSS keeps BM25 + fallback
• Providers are re-ranked locally for consistency

This turns WebAgent into a true multi-source research engine.
"""

from __future__ import annotations

import asyncio
import datetime
import logging
import math
import random
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime as dt, timezone
from email.utils import parsedate_to_datetime
from typing import List, Optional

from cognition.research_mcp.schemas import WebPage
from cognition.research_mcp.search_providers import (
    SearchConfig, get_results, _cred, _is_blocked, _STEALTH_JS, _UAS
)

logger = logging.getLogger("research_mcp.web_agent")

_MAX_CONTENT_CHARS = 8000

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONFIG
# ─────────────────────────────────────────────────────────────────────────────
_CFG: SearchConfig = SearchConfig()

# Source reliability priors
_SOURCE_WEIGHT = {
    "rss": 0.6,
    "brave": 0.8,
    "serpapi": 0.85,
    "claude": 0.9,
    "unknown": 0.5
}

# ─────────────────────────────────────────────────────────────────────────────
# TOKENIZATION + BM25
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str):
    return [t.lower() for t in text.split() if len(t) > 2]


def _bm25_score(query_terms, doc_terms, avgdl, k1=1.5, b=0.75):
    if not doc_terms:
        return 0.0

    tf = Counter(doc_terms)
    doc_len = len(doc_terms)

    score = 0.0
    for term in query_terms:
        if term not in tf:
            continue

        f = tf[term]
        idf = math.log(1 + (1 / (1 + f)))
        denom = f + k1 * (1 - b + b * (doc_len / avgdl))
        score += idf * ((f * (k1 + 1)) / denom)

    return score


def _recency_score(pub_date: dt | None):
    if not pub_date:
        return 0.0

    now = dt.now(timezone.utc)
    age_hours = (now - pub_date).total_seconds() / 3600
    return math.exp(-age_hours / 48)


def _parse_date(date_str: str):
    try:
        return parsedate_to_datetime(date_str)
    except Exception:
        return None

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL SCORING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def _global_score(query_terms, text, credibility, source, pub_date=None):
    terms = _tokenize(text)
    avgdl = max(len(terms), 1)

    relevance = _bm25_score(query_terms, terms, avgdl)
    recency = _recency_score(pub_date)
    source_w = _SOURCE_WEIGHT.get(source, _SOURCE_WEIGHT["unknown"])

    return (relevance * 0.5) + (recency * 0.2) + (credibility * 0.2) + (source_w * 0.1)

# ─────────────────────────────────────────────────────────────────────────────
# RSS ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def _rss(query: str, feeds: list[str], num: int = 50):
    query_terms = _tokenize(query)
    items = []

    import requests

    for feed in feeds:
        try:
            r = requests.get(feed, timeout=5)
            root = ET.fromstring(r.content)

            for node in root.findall(".//item")[:100]:
                title = node.findtext("title", "")
                desc = node.findtext("description", "")
                link = node.findtext("link", "")
                pub_date = _parse_date(node.findtext("pubDate", ""))

                text = f"{title} {desc}"

                items.append({
                    "title": title,
                    "snippet": desc,
                    "url": link,
                    "text": text,
                    "pub_date": pub_date,
                    "source": "rss",
                    "credibility": _cred(link)
                })

        except Exception:
            continue

    if not items:
        return []

    scored = []
    for it in items:
        score = _global_score(query_terms, it["text"], it["credibility"], "rss", it["pub_date"])
        scored.append((score, it))

    if all(s[0] < 0.01 for s in scored):
        scored = [( _recency_score(i["pub_date"]), i ) for _, i in scored]

    scored.sort(key=lambda x: x[0], reverse=True)

    return [{
        "title": i["title"],
        "url": i["url"],
        "snippet": i["snippet"],
        "score": s,
        "source": "rss",
        "credibility": i["credibility"]
    } for s, i in scored[:num]]

# ─────────────────────────────────────────────────────────────────────────────
# FETCH LAYER
# ─────────────────────────────────────────────────────────────────────────────

try:
    from playwright.async_api import async_playwright as _apw
    _HAS_PLAYWRIGHT = True
except ImportError:
    _HAS_PLAYWRIGHT = False


async def _async_fetch(url: str):
    async with _apw() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=random.choice(_UAS))
        await context.add_init_script(_STEALTH_JS)
        page = await context.new_page()

        try:
            await page.goto(url, timeout=15000)
            title = await page.title()
            content = await page.evaluate("document.body.innerText")
            return title, content, ""
        except Exception as e:
            return "", "", str(e)
        finally:
            await browser.close()


def _sync_fetch(url: str):
    if _HAS_PLAYWRIGHT:
        try:
            return asyncio.run(_async_fetch(url))
        except Exception:
            pass

    try:
        import requests, re
        r = requests.get(url, timeout=10)
        text = re.sub(r'<[^>]+>', ' ', r.text)
        return url, text, ""
    except Exception as e:
        return "", "", str(e)

# ─────────────────────────────────────────────────────────────────────────────
# WEB AGENT
# ─────────────────────────────────────────────────────────────────────────────

class WebAgent:
    MAX_PAGES_PER_SESSION = 20

    def __init__(self, cfg: Optional[SearchConfig] = None):
        self._cfg = cfg or globals().get('_CFG', SearchConfig())
        self._pages_visited = 0
        self._visited_urls = set()

    def search_and_fetch(self, query: str, max_pages: int = 3):
        query_terms = _tokenize(query)

        results = get_results(query, cfg=self._cfg, num=max_pages * 3)

        # Re-rank ALL provider results
        rescored = []
        for item in results:
            url = item.get("url")
            if not url or _is_blocked(url):
                continue

            text = f"{item.get('title','')} {item.get('snippet','')}"
            source = item.get("source", "unknown")
            credibility = item.get("credibility", _cred(url))

            score = _global_score(query_terms, text, credibility, source)
            rescored.append((score, item))

        rescored.sort(key=lambda x: x[0], reverse=True)

        pages: List[WebPage] = []

        for score, item in rescored:
            if len(pages) >= max_pages:
                break

            url = item.get("url")
            content = item.get("content", "")

            if len(content) > 500 and item.get("source") in ("claude", "brave", "serpapi"):
                pages.append(WebPage(
                    url=url,
                    title=item.get("title", ""),
                    content=content[:_MAX_CONTENT_CHARS],
                    fetched_at=datetime.datetime.now().isoformat(),
                    credibility=item.get("credibility", _cred(url)),
                    source=item.get("source", "unknown"),
                ))
                continue

            title, content, error = _sync_fetch(url)

            if not error:
                pages.append(WebPage(
                    url=url,
                    title=title,
                    content=content[:_MAX_CONTENT_CHARS],
                    fetched_at=datetime.datetime.now().isoformat(),
                    credibility=_cred(url),
                    source=item.get("source", "unknown"),
                ))

        return pages


# ── Compatibility stub ────────────────────────────────────────────────────────
# Added for persona_bridge import compatibility
_search_backend = 'auto'

def set_search_backend(backend: str = 'auto', **kwargs) -> None:
    """Set the active search backend. Stub for compatibility — accepts any kwargs."""
    global _search_backend
    _search_backend = backend

def get_search_backend() -> str:
    return _search_backend
