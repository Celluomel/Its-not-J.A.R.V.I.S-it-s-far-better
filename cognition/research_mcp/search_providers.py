"""
search_providers.py  —  Three-tier resilient search for PandoraBOX Research MCP
=============================================================================

WHY THIS EXISTS
---------------
Search engines (DDG/Bing) actively block headless browsers.
Your testing confirmed: Playwright launches fine, page loads fine,
but DOM returns 0 results — bot-detection serving an empty/challenge page.

SOLUTION: Three tiers tried in order until results are found.

  TIER 1 — Structured APIs  (no browser, no bot risk, most reliable)
    A. Brave Search API   — free 2 000/month, sign up at api.search.brave.com
    B. Anthropic Claude   — built-in web_search tool (needs ANTHROPIC_API_KEY)
    C. SerpAPI            — if SERPAPI_KEY configured

  TIER 2 — Direct RSS / JSON feeds  (zero bot risk, great for news)
    BBC, Reuters, AP, Guardian, Hacker News, NYT, WSJ RSS
    Always works, no API key, perfect for "latest news" queries

  TIER 3 — Stealth browser scraping  (last resort, arms-race but resilient)
    DuckDuckGo + Bing with:
    - playwright-stealth JS injection (navigator.webdriver spoofed)
    - Random UA, viewport, timing
    - Human-like scroll before extraction
    - Multiple selector fallback strategies

CONFIG (set via /llm settings page or config.json):
  RESEARCH_SEARCH_BACKEND: "auto" | "claude" | "brave" | "rss" | "stealth"
  ANTHROPIC_API_KEY, BRAVE_SEARCH_KEY, SERPAPI_KEY
"""
from __future__ import annotations

import asyncio
import logging
import random
from typing import List, Optional
from urllib.parse import urlparse, quote_plus

logger = logging.getLogger("research_mcp.search_providers")

# ── Credibility scoring ───────────────────────────────────────────────────────
_HIGH = {
    "nature.com","science.org","arxiv.org","bbc.com","bbc.co.uk","reuters.com",
    "apnews.com","theguardian.com","nytimes.com","wsj.com","ft.com",
    "economist.com","bloomberg.com","npr.org","pbs.org","mit.edu","stanford.edu",
    "harvard.edu","nih.gov","cdc.gov","who.int","nasa.gov","wikipedia.org",
    "techcrunch.com","wired.com","arstechnica.com","ieee.org","theatlantic.com",
    "axios.com","politico.com","foreignpolicy.com","ap.org",
}
_LOW = {
    "reddit.com","quora.com","twitter.com","x.com","tiktok.com",
    "facebook.com","instagram.com","pinterest.com",
}
_BLACKLIST = _LOW  # same set


def _cred(url: str) -> float:
    if not url:
        return 0.5
    d = urlparse(url).netloc.lower().lstrip("www.")
    if any(h in d for h in _HIGH):         return 0.88
    if any(l in d for l in _LOW):          return 0.28
    if d.endswith((".edu",".gov",".org")): return 0.72
    return 0.55


def _is_blocked(url: str) -> bool:
    d = urlparse(url).netloc.lower().lstrip("www.")
    return any(b in d for b in _BLACKLIST)


def _r(title: str, url: str, snippet: str = "", content: str = "", src: str = "") -> dict:
    return {
        "title":       (title or url)[:200],
        "url":         url,
        "snippet":     snippet[:400],
        "content":     content,
        "credibility": _cred(url),
        "source":      src,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  TIER 1A — Brave Search API
# ─────────────────────────────────────────────────────────────────────────────
def _brave(query: str, num: int, key: str) -> List[dict]:
    """
    Brave Search API — free tier 2 000 req/month.
    Get key at https://api.search.brave.com (2-minute signup).
    Falls back gracefully if key missing or quota hit.
    """
    if not key:
        return []
    try:
        import requests as _req
        r = _req.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": num, "text_decorations": False},
            headers={
                "Accept":                  "application/json",
                "Accept-Encoding":         "gzip",
                "X-Subscription-Token":    key,
            },
            timeout=8,
        )
        if r.status_code == 429:
            logger.warning("Brave API: rate limit hit")
            return []
        if r.status_code != 200:
            logger.warning(f"Brave API: HTTP {r.status_code}")
            return []
        items = r.json().get("web", {}).get("results", [])
        out = [_r(i.get("title",""), i.get("url",""), i.get("description",""), src="brave") for i in items[:num]]
        logger.info(f"Brave: {len(out)} results for {query!r}")
        return out
    except Exception as e:
        logger.warning(f"Brave failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  TIER 1B — Anthropic Claude web_search tool
# ─────────────────────────────────────────────────────────────────────────────
def _claude(query: str, num: int, key: str) -> List[dict]:
    """Uses Anthropic API with built-in web_search. Needs: pip install anthropic"""
    if not key:
        return []
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model      = "claude-haiku-4-5-20251001",
            max_tokens = 1500,
            tools      = [{"type": "web_search_20250305", "name": "web_search"}],
            messages   = [{"role": "user", "content":
                f"Search the web for: {query}\nReturn the {num} most relevant results."}],
        )
        out, seen = [], set()
        for block in resp.content:
            btype = getattr(block, "type", "")
            if btype == "web_search_result":
                url = getattr(block, "url", "")
                if url and url not in seen:
                    seen.add(url)
                    out.append(_r(getattr(block,"title",""), url,
                                  getattr(block,"snippet",""), getattr(block,"content",""),
                                  src="claude"))
            elif btype == "tool_result":
                for item in (block.content if isinstance(block.content, list) else []):
                    for cite in getattr(item, "citations", []):
                        url = getattr(cite, "url", "")
                        if url and url not in seen:
                            seen.add(url)
                            out.append(_r(getattr(cite,"title",""), url,
                                          getattr(item,"text","")[:300], src="claude"))
        logger.info(f"Claude search: {len(out)} results for {query!r}")
        return out[:num]
    except ImportError:
        logger.warning("Claude search: run 'pip install anthropic'")
        return []
    except Exception as e:
        logger.warning(f"Claude search failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  TIER 1C — SerpAPI
# ─────────────────────────────────────────────────────────────────────────────
def _serpapi(query: str, num: int, key: str) -> List[dict]:
    if not key:
        return []
    try:
        import requests as _req
        r = _req.get(
            "https://serpapi.com/search",
            params={"q": query, "num": num, "api_key": key, "engine": "google"},
            timeout=8,
        )
        items = r.json().get("organic_results", [])
        out = [_r(i.get("title",""), i.get("link",""), i.get("snippet",""), src="serpapi") for i in items[:num]]
        logger.info(f"SerpAPI: {len(out)} results for {query!r}")
        return out
    except Exception as e:
        logger.warning(f"SerpAPI failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  TIER 2 — RSS / JSON direct feeds  (zero bot risk)
#  Feed list is now configurable — see cognition/research_mcp/rss_registry.py.
#  Managed via the RSS Management page: activate/deactivate, tag by domain
#  (news/science/tech/...), and each feed's success rate is tracked from
#  real research session outcomes (Evaluator.assess()'s confidence_score).
# ─────────────────────────────────────────────────────────────────────────────


# ── Lightweight domain guess from query text ────────────────────────────────
# Self-contained on purpose: avoids threading a new parameter through the
# whole get_results()/SearchConfig call chain for what's meant to be a
# soft ranking hint, not a hard classification.
_TAG_KEYWORDS = {
    "science":  {"study", "research", "paper", "scientist", "physics", "biology",
                 "chemistry", "experiment", "arxiv", "peer-reviewed", "discovery"},
    "tech":     {"software", "ai", "algorithm", "programming", "startup", "app",
                 "hardware", "chip", "code", "github", "computer"},
    "news":     {"election", "government", "war", "economy", "politics", "president",
                 "breaking", "today", "yesterday"},
    "finance":  {"stock", "market", "economy", "inflation", "investment", "bank",
                 "trading", "crypto"},
    "health":   {"health", "medicine", "disease", "treatment", "vaccine", "hospital",
                 "doctor", "clinical"},
}


def _guess_tag_from_query(query: str) -> Optional[str]:
    words = set(query.lower().split())
    best_tag, best_score = None, 0
    for tag, keywords in _TAG_KEYWORDS.items():
        score = len(words & keywords)
        if score > best_score:
            best_tag, best_score = tag, score
    return best_tag


def _rss(query: str, num: int, tag_hint: Optional[str] = None) -> List[dict]:
    """Parse RSS feeds from the configurable registry, score relevance by
    keyword match. Always works. Feeds are active-only and ranked by tag
    relevance + historical success rate — see rss_registry.py."""
    try:
        import requests as _req
        import xml.etree.ElementTree as ET
        from cognition.research_mcp.rss_registry import get_rss_registry

        registry = get_rss_registry()
        if tag_hint is None:
            tag_hint = _guess_tag_from_query(query)
        feeds = registry.active_feeds(tag_hint=tag_hint)
        if not feeds:
            logger.info("RSS: no active feeds configured")
            return []

        terms = set(query.lower().split())
        candidates = []
        for feed in feeds:
            try:
                r = _req.get(feed.url, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                if r.status_code != 200:
                    continue
                root  = ET.fromstring(r.content)
                for item in root.findall(".//item")[:25]:
                    title = (item.findtext("title") or "").strip()
                    link  = (item.findtext("link")  or "").strip()
                    desc  = (item.findtext("description") or "").strip()
                    if not title or not link:
                        continue
                    score = sum(1 for t in terms if t in (title+" "+desc).lower()) / max(len(terms),1)
                    candidates.append((score, _r(title, link, desc[:300], src=f"rss_{feed.name}")))
                registry.mark_used(feed.feed_id)
            except Exception:
                continue
        if not candidates:
            return []
        candidates.sort(key=lambda x: (x[0], x[1]["credibility"]), reverse=True)
        out = [c[1] for c in candidates[:num]]
        logger.info(f"RSS: {len(out)} results for {query!r}")
        return out
    except Exception as e:
        logger.warning(f"RSS failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  TIER 3 — Stealth browser scraping
# ─────────────────────────────────────────────────────────────────────────────
_STEALTH_JS = """
Object.defineProperty(navigator,'webdriver',{get:()=>undefined});
Object.defineProperty(navigator,'plugins',  {get:()=>[1,2,3,4,5]});
Object.defineProperty(navigator,'languages',{get:()=>['en-US','en']});
Object.defineProperty(navigator,'platform', {get:()=>'Win32'});
Object.defineProperty(navigator,'hardwareConcurrency',{get:()=>8});
Object.defineProperty(navigator,'deviceMemory',       {get:()=>8});
window.chrome={runtime:{}};
const _oq=window.navigator.permissions.query.bind(navigator.permissions);
navigator.permissions.query=p=>p.name==='notifications'?Promise.resolve({state:Notification.permission}):_oq(p);
"""

_UAS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
]
_VIEWPORTS = [
    {"width":1920,"height":1080},
    {"width":1366,"height":768},
    {"width":1536,"height":864},
]

_DDG_CONTAINERS = [
    'article[data-testid="result"]',
    'li[data-layout="organic"]',
    'div[data-nrn="result"]',
    'section[data-testid="mainline"] article',
    'div.nrn-react-div article',
    '#links .result',
    'li.PartialSearchResults-item',
]
_DDG_LINKS = ['a[data-testid="result-title-a"]','h2 > a','a.result__a','.result__title a']
_DDG_SNIPS = ['div[data-testid="result-snippet"]','.result__snippet']


async def _make_stealth_context(pw):
    """Create a stealth-patched browser context."""
    browser = await pw.chromium.launch(
        headless=True,
        args=[
            "--disable-blink-features=AutomationControlled",
            "--disable-dev-shm-usage",
            "--no-sandbox",
            "--disable-web-security",
            "--disable-features=IsolateOrigins,site-per-process",
        ],
    )
    context = await browser.new_context(
        user_agent         = random.choice(_UAS),
        viewport           = random.choice(_VIEWPORTS),
        locale             = "en-US",
        timezone_id        = "America/New_York",
        extra_http_headers = {"Accept-Language":"en-US,en;q=0.9",
                              "Accept":"text/html,application/xhtml+xml,*/*;q=0.8"},
    )
    await context.add_init_script(_STEALTH_JS)

    # playwright-stealth plugin (optional, improves success rate)
    try:
        from playwright_stealth import stealth_async as _sa
        context._stealth_fn = _sa
    except ImportError:
        context._stealth_fn = None

    async def _block(route):
        if route.request.resource_type in ("image","font","media","stylesheet"):
            await route.abort()
        else:
            await route.continue_()
    await context.route("**/*", _block)
    return browser, context


async def _extract_ddg(page, num: int) -> list:
    return await page.evaluate("""
        (args) => {
            const {cs,ls,ss,num} = args;
            function find(p,sels){for(const s of sels){const e=p.querySelector(s);if(e)return e;}return null;}
            for(const c of cs){
                const els=document.querySelectorAll(c);
                if(!els.length)continue;
                const items=[];
                for(const el of els){
                    const a=find(el,ls), sn=find(el,ss);
                    const url=a?.href||'', title=(a?.innerText||'').trim();
                    if(url&&title&&!url.includes('duckduckgo.com'))
                        items.push({title,url,snippet:(sn?.innerText||'').trim()});
                    if(items.length>=num)break;
                }
                if(items.length)return items;
            }
            // fallback: any outbound link with text
            const seen=new Set(),fb=[];
            for(const a of document.querySelectorAll('a[href^="http"]')){
                const h=a.href,t=(a.innerText||'').trim();
                if(t.length>20&&!h.includes('duckduckgo.com')&&!seen.has(h)){
                    seen.add(h);fb.push({title:t,url:h,snippet:''});
                    if(fb.length>=num)break;
                }
            }
            return fb;
        }
    """, {"cs": _DDG_CONTAINERS, "ls": _DDG_LINKS, "ss": _DDG_SNIPS, "num": num})


async def _stealth_ddg(query: str, num: int) -> List[dict]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return []
    async with async_playwright() as pw:
        browser, context = await _make_stealth_context(pw)
        page = await context.new_page()
        if context._stealth_fn:
            await context._stealth_fn(page)
        try:
            await asyncio.sleep(random.uniform(0.5,1.2))
            await page.goto(f"https://duckduckgo.com/?q={quote_plus(query)}&kp=-1",
                            timeout=18000, wait_until="domcontentloaded")
            # Human-like: scroll, wait
            await asyncio.sleep(random.uniform(1.5,2.8))
            await page.mouse.move(random.randint(200,600), random.randint(200,400), steps=12)
            await page.mouse.wheel(0, random.randint(300,600))
            await asyncio.sleep(random.uniform(0.8,1.5))

            raw = await _extract_ddg(page, num+2)
            raw = [r for r in raw if r.get("url","").startswith("http") and not _is_blocked(r["url"])]
            out = [_r(r["title"],r["url"],r.get("snippet",""),src="ddg_stealth") for r in raw[:num]]
            logger.info(f"Stealth DDG: {len(out)} results for {query!r}")
            return out
        except Exception as e:
            logger.warning(f"Stealth DDG error: {e}")
            return []
        finally:
            await page.close()
            await context.close()
            await browser.close()


async def _stealth_bing(query: str, num: int) -> List[dict]:
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        return []
    async with async_playwright() as pw:
        browser, context = await _make_stealth_context(pw)
        page = await context.new_page()
        if context._stealth_fn:
            await context._stealth_fn(page)
        try:
            await asyncio.sleep(random.uniform(0.4,1.0))
            await page.goto(f"https://www.bing.com/search?q={quote_plus(query)}",
                            timeout=18000, wait_until="domcontentloaded")
            await asyncio.sleep(random.uniform(1.5,2.5))
            await page.mouse.wheel(0, random.randint(200,500))
            await asyncio.sleep(random.uniform(0.5,1.0))

            raw = await page.evaluate("""
                (num) => {
                    const items=[];
                    for(const li of document.querySelectorAll('li.b_algo')){
                        const a=li.querySelector('h2 a');
                        const sn=li.querySelector('.b_caption p');
                        if(a?.href&&(a?.innerText||'').trim())
                            items.push({title:a.innerText.trim(),url:a.href,
                                        snippet:(sn?.innerText||'').trim()});
                        if(items.length>=num)break;
                    }
                    return items;
                }
            """, num)
            out = [_r(r["title"],r["url"],r.get("snippet",""),src="bing_stealth") for r in (raw or [])]
            logger.info(f"Stealth Bing: {len(out)} results for {query!r}")
            return out
        except Exception as e:
            logger.warning(f"Stealth Bing error: {e}")
            return []
        finally:
            await page.close()
            await context.close()
            await browser.close()


def _stealth(query: str, num: int) -> List[dict]:
    """Run stealth DDG + Bing sequentially (sync wrapper)."""
    try:
        out = asyncio.run(_stealth_ddg(query, num))
        if not out:
            logger.info("DDG stealth empty → trying Bing stealth")
            out = asyncio.run(_stealth_bing(query, num))
        return out
    except Exception as e:
        logger.error(f"Stealth search failed: {e}")
        return []


# ─────────────────────────────────────────────────────────────────────────────
#  SearchConfig  (passed in from WebAgent / settings)
# ─────────────────────────────────────────────────────────────────────────────
class SearchConfig:
    def __init__(self,
                 backend:       str  = "auto",
                 anthropic_key: str  = "",
                 brave_key:     str  = "",
                 serpapi_key:   str  = "",
                 use_rss:       bool = True,
                 use_stealth:   bool = True,
                 num_results:   int  = 5):
        self.backend       = backend
        self.anthropic_key = anthropic_key
        self.brave_key     = brave_key
        self.serpapi_key   = serpapi_key
        self.use_rss       = use_rss
        self.use_stealth   = use_stealth
        self.num_results   = num_results


# ─────────────────────────────────────────────────────────────────────────────
#  PUBLIC ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def get_results(query: str, cfg: Optional[SearchConfig] = None, num: int = 5) -> List[dict]:
    """
    Master search — tries tiers in order, merges, deduplicates, returns top N.
    Sync: runs async tiers via asyncio.run() (safe from threadpool threads).
    """
    if cfg is None:
        cfg = SearchConfig()
    num = cfg.num_results or num

    all_results: List[dict] = []
    seen:        set        = set()

    def _add(items):
        for r in items:
            if r.get("url") and r["url"] not in seen:
                seen.add(r["url"])
                all_results.append(r)

    b = cfg.backend

    # ── Explicit single backend ───────────────────────────────────────────────
    if b == "claude":
        _add(_claude(query, num, cfg.anthropic_key))
    elif b == "brave":
        _add(_brave(query, num, cfg.brave_key))
    elif b == "rss":
        _add(_rss(query, num))
    elif b == "stealth":
        _add(_stealth(query, num))

    # ── AUTO: all tiers in priority order ────────────────────────────────────
    else:
        # T1A Brave
        if cfg.brave_key and len(all_results) < num:
            _add(_brave(query, num, cfg.brave_key))

        # T1B Claude
        if cfg.anthropic_key and len(all_results) < num:
            _add(_claude(query, num, cfg.anthropic_key))

        # T1C SerpAPI
        if cfg.serpapi_key and len(all_results) < num:
            _add(_serpapi(query, num, cfg.serpapi_key))

        # T2 RSS (great for news, always works)
        if cfg.use_rss and len(all_results) < num:
            _add(_rss(query, max(num - len(all_results), 3)))

        # T3 Stealth (last resort)
        if cfg.use_stealth and len(all_results) < 2:
            logger.info("All API/RSS tiers exhausted — falling back to stealth browser")
            _add(_stealth(query, num))

    all_results.sort(key=lambda x: x["credibility"], reverse=True)
    final = all_results[:num]
    srcs  = {r["source"] for r in final}
    logger.info(f"search_providers: {len(final)} results for {query!r} via {srcs}")
    return final
