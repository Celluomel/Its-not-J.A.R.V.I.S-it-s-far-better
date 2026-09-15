"""
Claude Search Backend — uses Anthropic's native web_search tool.

Advantages over Playwright/DDG:
  - No browser required, no DOM scraping, no selector rot
  - Returns pre-extracted clean text (not raw HTML)
  - Works anywhere with an internet connection + API key
  - Anthropic handles anti-bot, CAPTCHA, rate limits

Requires:
  - ANTHROPIC_API_KEY in config.json (Settings → LLM tab → Anthropic API Key)
  - pip install anthropic  (lightweight, no browser)

Usage:
  results = claude_search(query="latest AI news", num_results=5, api_key="sk-ant-...")
  → List[{title, url, snippet, content, credibility}]
"""
from __future__ import annotations
import json
import logging
from typing import List, Optional
from urllib.parse import urlparse

logger = logging.getLogger("research_mcp.claude_search")

# ── Credibility (reused from web_agent) ──────────────────────────────────────
_HIGH_CRED = {
    "nature.com","science.org","arxiv.org","bbc.com","reuters.com",
    "apnews.com","theguardian.com","nytimes.com","wsj.com","mit.edu",
    "stanford.edu","harvard.edu","nih.gov","cdc.gov","who.int","nasa.gov",
    "wikipedia.org","techcrunch.com","wired.com","arstechnica.com","ieee.org",
}
_LOW_CRED = {"reddit.com","quora.com","twitter.com","x.com","medium.com"}


def _credibility(url: str) -> float:
    domain = urlparse(url).netloc.lower().lstrip("www.")
    if any(h in domain for h in _HIGH_CRED):    return 0.85
    if any(l in domain for l in _LOW_CRED):     return 0.38
    if domain.endswith((".edu",".gov",".org")):  return 0.70
    return 0.58


# ── Check availability ────────────────────────────────────────────────────────
def is_available(api_key: str) -> bool:
    """Returns True if anthropic package is installed and key is non-empty."""
    if not api_key or not api_key.strip():
        return False
    try:
        import anthropic  # noqa
        return True
    except ImportError:
        return False


# ── Main search function ──────────────────────────────────────────────────────
def claude_search(
    query:       str,
    num_results: int = 5,
    api_key:     str = "",
    model:       str = "claude-haiku-4-5-20251001",   # cheapest model for search
) -> List[dict]:
    """
    Use Claude's web_search tool to search the web.
    Returns List[{title, url, snippet, content, credibility}].

    Uses claude-haiku (fastest/cheapest) by default since this is just search,
    not generation. The research system's own LLM handles synthesis.
    """
    if not api_key:
        logger.error("Claude search: no ANTHROPIC_API_KEY configured")
        return []

    try:
        import anthropic
    except ImportError:
        logger.error("Claude search: 'anthropic' package not installed. Run: pip install anthropic")
        return []

    client = anthropic.Anthropic(api_key=api_key)

    prompt = (
        f"Search the web for: {query}\n\n"
        f"Find the {num_results} most relevant, recent results. "
        f"For each result provide the title, URL, and a brief summary of what the page contains."
    )

    try:
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )

        results = _parse_response(response, num_results)
        logger.info(f"Claude search: {len(results)} results for {query!r}")
        return results

    except anthropic.AuthenticationError:
        logger.error("Claude search: invalid API key")
        return []
    except anthropic.RateLimitError:
        logger.warning("Claude search: rate limit hit")
        return []
    except Exception as e:
        logger.error(f"Claude search error: {e}")
        return []


def _parse_response(response, num_results: int) -> List[dict]:
    """
    Parse Anthropic response content blocks.
    web_search returns tool_result blocks with citations + text blocks with summary.
    """
    results = []
    seen_urls = set()

    for block in response.content:
        # Tool result blocks contain the actual search results with citations
        if block.type == "tool_result":
            try:
                for item in (block.content if isinstance(block.content, list) else []):
                    if hasattr(item, "citations"):
                        for cite in item.citations:
                            url   = getattr(cite, "url", "")
                            title = getattr(cite, "title", "") or url
                            if url and url not in seen_urls:
                                seen_urls.add(url)
                                results.append({
                                    "title":       title,
                                    "url":         url,
                                    "snippet":     getattr(item, "text", "")[:300],
                                    "content":     getattr(item, "text", ""),
                                    "credibility": _credibility(url),
                                })
            except Exception:
                pass

        # Text blocks contain Claude's synthesis — extract any URLs mentioned
        elif block.type == "text":
            _extract_urls_from_text(block.text, results, seen_urls)

        # web_search_result blocks (newer API versions)
        elif block.type == "web_search_result":
            try:
                url   = getattr(block, "url", "")
                title = getattr(block, "title", "") or url
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    results.append({
                        "title":       title,
                        "url":         url,
                        "snippet":     getattr(block, "snippet", "")[:300],
                        "content":     getattr(block, "content", ""),
                        "credibility": _credibility(url),
                    })
            except Exception:
                pass

    # If no structured results parsed, create one entry from the text summary
    if not results:
        text_blocks = [b.text for b in response.content if hasattr(b, "text") and b.text]
        if text_blocks:
            combined = " ".join(text_blocks)
            results.append({
                "title":       f"Search results for: {combined[:60]}",
                "url":         "",
                "snippet":     combined[:500],
                "content":     combined,
                "credibility": 0.70,
            })

    return results[:num_results]


def _extract_urls_from_text(text: str, results: list, seen: set):
    """Extract any markdown-linked URLs from text blocks as a fallback."""
    import re
    for match in re.finditer(r'\[([^\]]+)\]\((https?://[^\)]+)\)', text):
        title, url = match.group(1), match.group(2)
        if url not in seen:
            seen.add(url)
            results.append({
                "title":       title,
                "url":         url,
                "snippet":     "",
                "content":     "",
                "credibility": _credibility(url),
            })
