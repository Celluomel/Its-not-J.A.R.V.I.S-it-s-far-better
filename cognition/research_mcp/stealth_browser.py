"""Web Agent for PandoraBOX - Real DuckDuckGo Search with Stealth Mode"""
import asyncio
import random
import time
import logging
from urllib.parse import quote_plus, urlparse
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

from playwright.async_api import async_playwright, Browser, Page, Route

logger = logging.getLogger(__name__)

# ============================================================================
# STEALTH CONFIGURATION - Makes headless browser look human
# ============================================================================

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
]

VIEWPORTS = [
    {"width": 1920, "height": 1080},
    {"width": 1366, "height": 768},
    {"width": 1536, "height": 864}
]

# ============================================================================
# DOMAIN CREDIBILITY SCORING
# ============================================================================

DOMAIN_SCORES = {
    # Academic/Research (high credibility)
    "wikipedia.org": 0.85,
    "arxiv.org": 0.95,
    "nature.com": 0.95,
    "science.org": 0.95,
    "nih.gov": 0.95,
    "edu": 0.90,
    "ac.uk": 0.90,
    "ac.jp": 0.90,
    "cambridge.org": 0.92,
    "springer.com": 0.92,
    "ieee.org": 0.92,
    "acm.org": 0.92,
    
    # Tech/News (medium-high)
    "techcrunch.com": 0.75,
    "wired.com": 0.80,
    "arstechnica.com": 0.80,
    "theverge.com": 0.75,
    "zdnet.com": 0.70,
    "cnet.com": 0.70,
    
    # General knowledge (medium)
    "britannica.com": 0.85,
    "bbc.com": 0.82,
    "reuters.com": 0.85,
    "apnews.com": 0.85,
    "nytimes.com": 0.80,
    "wsj.com": 0.82,
    "theguardian.com": 0.78,
    
    # Developer resources (medium-high)
    "github.com": 0.80,
    "stackoverflow.com": 0.75,
    "gitlab.com": 0.75,
    "medium.com": 0.55,  # User-generated, variable quality
    "dev.to": 0.60,
    
    # Lower credibility
    "reddit.com": 0.40,
    "quora.com": 0.45,
    "youtube.com": 0.50,
    "twitter.com": 0.30,
    "x.com": 0.30,
    "tiktok.com": 0.20,
    "facebook.com": 0.25,
    "instagram.com": 0.20,
}

DEFAULT_CREDIBILITY = 0.60  # Neutral default

def score_domain(url: str) -> float:
    """Score domain credibility based on known patterns"""
    try:
        domain = urlparse(url).netloc.lower()
        domain = domain.replace("www.", "")
        
        # Check exact matches
        if domain in DOMAIN_SCORES:
            return DOMAIN_SCORES[domain]
        
        # Check domain endings
        for pattern, score in DOMAIN_SCORES.items():
            if domain.endswith(pattern):
                return score
        
        return DEFAULT_CREDIBILITY
    except Exception:
        return DEFAULT_CREDIBILITY

# ============================================================================
# CORE BROWSER CLASS
# ============================================================================

class StealthBrowser:
    """Persistent headless browser with stealth mode and resource optimization"""
    
    def __init__(self):
        self.playwright = None
        self.browser: Optional[Browser] = None
        self.context = None
        self.initialized = False
        self._lock = asyncio.Lock()
        
    async def initialize(self):
        """Initialize browser with stealth configuration"""
        async with self._lock:
            if self.initialized:
                return
            
            logger.info("🚀 Initializing stealth browser...")
            
            self.playwright = await async_playwright().start()
            
            # Launch with anti-detection args
            self.browser = await self.playwright.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--disable-dev-shm-usage",
                    "--no-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                    "--disable-site-isolation-trials",
                ]
            )
            
            # Create context with human-like properties
            self.context = await self.browser.new_context(
                user_agent=random.choice(USER_AGENTS),
                viewport=random.choice(VIEWPORTS),
                locale="en-US,en;q=0.9",
                timezone_id="America/New_York",
                permissions=["geolocation"],
                device_scale_factor=1,
                has_touch=False,
                accept_downloads=False,
                extra_http_headers={
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Connection": "keep-alive",
                    "Upgrade-Insecure-Requests": "1",
                }
            )
            
            # Block unnecessary resources for speed
            await self.context.route("**/*", self._block_resources)
            
            # Inject stealth JavaScript
            await self.context.add_init_script("""
                // Remove webdriver property
                Object.defineProperty(navigator, 'webdriver', {
                    get: () => undefined
                });
                
                // Add missing plugins
                Object.defineProperty(navigator, 'plugins', {
                    get: () => [1, 2, 3, 4, 5]
                });
                
                // Add languages
                Object.defineProperty(navigator, 'languages', {
                    get: () => ['en-US', 'en']
                });
                
                // Add platform
                Object.defineProperty(navigator, 'platform', {
                    get: () => 'Win32'
                });
                
                // Add hardware concurrency
                Object.defineProperty(navigator, 'hardwareConcurrency', {
                    get: () => 8
                });
                
                // Add device memory
                Object.defineProperty(navigator, 'deviceMemory', {
                    get: () => 8
                });
                
                // Override chrome property
                window.chrome = {
                    runtime: {}
                };
                
                // Override permissions
                const originalQuery = window.navigator.permissions.query;
                window.navigator.permissions.query = (parameters) => (
                    parameters.name === 'notifications' ?
                        Promise.resolve({ state: Notification.permission }) :
                        originalQuery(parameters)
                );
            """)
            
            self.initialized = True
            logger.info("✅ Stealth browser ready")
    
    async def _block_resources(self, route: Route):
        """Block images, fonts, media for speed"""
        resource_type = route.request.resource_type
        if resource_type in ["image", "font", "media", "stylesheet"]:
            await route.abort()
        else:
            await route.continue_()
    
    async def random_delay(self, min_ms: int = 300, max_ms: int = 1200):
        """Add random human-like delay"""
        await asyncio.sleep(random.uniform(min_ms / 1000, max_ms / 1000))
    
    async def search_duckduckgo(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Perform real DuckDuckGo search and return structured results
        Returns list of {title, url, snippet}
        """
        await self.initialize()
        
        logger.info(f"🔍 Searching DuckDuckGo: {query}")
        
        # Create a new page for this search
        page = await self.context.new_page()
        
        try:
            # Add small random delay to appear human
            await self.random_delay()
            
            # Navigate to DuckDuckGo
            search_url = f"https://duckduckgo.com/?q={quote_plus(query)}&kp=-1"  # kp=-1 = no safe search
            await page.goto(search_url, timeout=15000, wait_until="domcontentloaded")
            
            # Wait for results to load
            await page.wait_for_selector('article[data-testid="result"]', timeout=10000)
            await self.random_delay(500, 1000)
            
            # Extract results
            results = await page.evaluate("""
                () => {
                    const items = [];
                    const resultElements = document.querySelectorAll('article[data-testid="result"]');
                    
                    for (const el of resultElements) {
                        try {
                            // Title and link
                            const titleEl = el.querySelector('a[data-testid="result-title-a"]');
                            const title = titleEl?.innerText || '';
                            const url = titleEl?.href || '';
                            
                            // Snippet
                            const snippetEl = el.querySelector('div[data-testid="result-snippet"]');
                            const snippet = snippetEl?.innerText || '';
                            
                            // Visible URL
                            const urlEl = el.querySelector('span[data-testid="result-url"]');
                            const displayUrl = urlEl?.innerText || '';
                            
                            if (title && url) {
                                items.push({
                                    title: title.trim(),
                                    url: url,
                                    snippet: snippet.trim(),
                                    displayUrl: displayUrl.trim()
                                });
                            }
                        } catch (e) {
                            // Skip malformed results
                        }
                    }
                    
                    return items.slice(0, 10);
                }
            """)
            
            # Add credibility scores
            for result in results:
                result["credibility"] = score_domain(result["url"])
                result["source"] = "duckduckgo"
            
            logger.info(f"✅ Found {len(results)} results for '{query}'")
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
            
        finally:
            await page.close()
    
    async def fetch_page(self, url: str, max_chars: int = 8000) -> Dict[str, Any]:
        """
        Fetch and extract content from a single page
        Returns {url, title, content, credibility}
        """
        await self.initialize()
        
        logger.info(f"📄 Fetching: {url}")
        
        page = await self.context.new_page()
        
        try:
            # Small random delay
            await self.random_delay()
            
            # Navigate with timeout
            await page.goto(url, timeout=15000, wait_until="domcontentloaded")
            await page.wait_for_load_state("networkidle", timeout=5000)
            
            # Get page info
            title = await page.title()
            
            # Extract main content (prioritize article/main tags)
            content = await page.evaluate("""
                () => {
                    // Try to find main content
                    const selectors = [
                        'article',
                        'main',
                        '[role="main"]',
                        '.content',
                        '#content',
                        '.post-content',
                        '.article-content',
                        'body'
                    ];
                    
                    for (const selector of selectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            // Get text, clean it up
                            let text = element.innerText || '';
                            // Remove extra whitespace
                            text = text.replace(/\\s+/g, ' ').trim();
                            if (text.length > 500) return text;
                        }
                    }
                    
                    // Fallback to body
                    return document.body.innerText.replace(/\\s+/g, ' ').trim();
                }
            """)
            
            # Truncate if needed
            if len(content) > max_chars:
                content = content[:max_chars] + "..."
            
            credibility = score_domain(url)
            
            logger.info(f"✅ Fetched {len(content)} chars from {title}")
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "credibility": credibility,
                "success": True
            }
            
        except Exception as e:
            logger.error(f"Fetch error for {url}: {e}")
            return {
                "url": url,
                "error": str(e),
                "success": False,
                "credibility": score_domain(url)
            }
            
        finally:
            await page.close()
    
    async def fetch_multiple(self, urls: List[str], max_concurrent: int = 3, max_chars: int = 6000) -> List[Dict[str, Any]]:
        """
        Fetch multiple pages in parallel with concurrency limit
        """
        await self.initialize()
        
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_one(url: str) -> Dict[str, Any]:
            async with semaphore:
                return await self.fetch_page(url, max_chars)
        
        tasks = [fetch_one(url) for url in urls]
        return await asyncio.gather(*tasks)
    
    async def research_topic(self, query: str, depth: int = 3) -> Dict[str, Any]:
        """
        Complete research flow: search → fetch top results → return structured data
        """
        # Step 1: Search
        search_results = await self.search_duckduckgo(query, max_results=depth + 2)
        
        if not search_results:
            return {
                "query": query,
                "error": "No search results found",
                "results": []
            }
        
        # Step 2: Filter by credibility (optional)
        filtered_results = [
            r for r in search_results 
            if r.get("credibility", 0.5) >= 0.4  # Filter out very low credibility
        ]
        
        # Step 3: Fetch top results in parallel
        urls_to_fetch = [r["url"] for r in filtered_results[:depth]]
        page_contents = await self.fetch_multiple(urls_to_fetch)
        
        # Step 4: Combine results
        combined_results = []
        for search_result, page_content in zip(filtered_results[:depth], page_contents):
            combined_results.append({
                "title": search_result["title"],
                "url": search_result["url"],
                "snippet": search_result.get("snippet", ""),
                "credibility": search_result.get("credibility", 0.6),
                "full_content": page_content.get("content", "") if page_content.get("success") else None,
                "fetch_error": page_content.get("error") if not page_content.get("success") else None
            })
        
        return {
            "query": query,
            "timestamp": time.time(),
            "result_count": len(combined_results),
            "results": combined_results
        }
    
    async def close(self):
        """Clean shutdown"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
        self.initialized = False
        logger.info("🛑 Browser closed")

# ============================================================================
# GLOBAL BROWSER INSTANCE (SINGLETON)
# ============================================================================

_browser_instance: Optional[StealthBrowser] = None

async def get_browser() -> StealthBrowser:
    """Get or create global browser instance"""
    global _browser_instance
    if _browser_instance is None:
        _browser_instance = StealthBrowser()
        await _browser_instance.initialize()
    return _browser_instance

# ============================================================================
# PUBLIC API - REPLACE THE STUBS IN YOUR ORIGINAL web_agent.py
# ============================================================================

async def _search(query: str, max_results: int = 5):
    """
    PUBLIC API: DuckDuckGo search
    Replace your existing stub with this
    """
    browser = await get_browser()
    return await browser.search_duckduckgo(query, max_results)


async def _fetch_url(url: str, max_chars: int = 6000):
    """
    PUBLIC API: Fetch single URL
    Replace your existing stub with this
    """
    browser = await get_browser()
    return await browser.fetch_page(url, max_chars)


async def _fetch_multiple(urls: list, max_concurrent: int = 3):
    """
    PUBLIC API: Fetch multiple URLs in parallel
    Useful for Mode 2/3 research
    """
    browser = await get_browser()
    return await browser.fetch_multiple(urls, max_concurrent)


async def _research_topic(query: str, depth: int = 3):
    """
    PUBLIC API: Complete research flow (search + fetch)
    Returns structured research data
    """
    browser = await get_browser()
    return await browser.research_topic(query, depth)


async def _close_browser():
    """Clean shutdown - call when app exits"""
    global _browser_instance
    if _browser_instance:
        await _browser_instance.close()
        _browser_instance = None

# ============================================================================
# SAMPLE USAGE (for testing)
# ============================================================================

if __name__ == "__main__":
    async def test():
        # Test search
        print("🔍 Testing search...")
        results = await _search("quantum computing breakthroughs 2025")
        for r in results:
            print(f"  • {r['title']} (cred: {r.get('credibility', 0):.2f})")
        
        # Test fetch
        if results:
            print("\n📄 Testing fetch...")
            content = await _fetch_url(results[0]["url"])
            print(f"  Title: {content.get('title')}")
            print(f"  Content length: {len(content.get('content', ''))}")
        
        await _close_browser()
    
    asyncio.run(test())