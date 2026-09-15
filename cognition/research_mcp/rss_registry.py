"""
cognition/research_mcp/rss_registry.py

Configurable RSS feed registry, replacing the hardcoded _FEEDS list that
used to live directly in search_providers.py. Each feed is:

  - activatable (checkbox — inactive feeds are never queried)
  - taggable (editable tags like "news", "science", "tech" — used to route
    a query toward feeds relevant to its domain, e.g. an arXiv feed tagged
    "science" gets preferred for a science-flavored query over a general
    news feed)
  - rated by real prior research success, not a static guess

Success/failure attribution: WebPage now carries a `source` field (e.g.
"rss_BBC") that survives from search_providers.py all the way into the
finished ResearchSession. ResearchController.run() already computes a
real per-session confidence_score from Evaluator.assess() — this
registry's record_session_outcome() reads which RSS-sourced pages
contributed to a session and updates each contributing feed's running
success rate from that session's actual confidence_score, rather than
inventing a new quality signal.

Persisted to data/persona/rss_feeds.json. Seeded on first run with the
feeds that used to be hardcoded in search_providers.py, all tagged
"news" (their original exclusive use), so existing behavior is
unchanged until the user edits anything via the management page.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("research_mcp.rss_registry")

DEFAULT_PATH = "data/persona/rss_feeds.json"

# Seed feeds — exactly what used to be hardcoded in search_providers.py's
# _FEEDS list, so nothing changes in behavior until the user edits the
# registry via the management page. All originally news-only, tagged
# accordingly; the user can retag/add (e.g. an arXiv feed tagged "science")
# through the UI.
_SEED_FEEDS = [
    ("https://feeds.bbci.co.uk/news/rss.xml",                "BBC",       ["news"]),
    ("https://feeds.reuters.com/reuters/topNews",             "Reuters",   ["news"]),
    ("https://feeds.theguardian.com/theguardian/world/rss",   "Guardian",  ["news"]),
    ("https://hnrss.org/frontpage",                           "HackerNews",["news", "tech"]),
    ("https://rss.nytimes.com/services/xml/rss/nyt/World.xml","NYT",       ["news"]),
    ("https://feeds.a.dj.com/rss/RSSWorldNews.xml",           "WSJ",       ["news"]),
    ("https://feeds.npr.org/1001/rss.xml",                    "NPR",       ["news"]),
    ("https://apnews.com/apf-topnews",                        "AP",        ["news"]),
    # Science feeds — added as the motivating example for tag-based routing.
    # arXiv publishes a real RSS feed per subject category.
    ("http://export.arxiv.org/rss/cs.AI",                     "arXiv-CS-AI",      ["science", "tech"]),
    ("http://export.arxiv.org/rss/physics",                   "arXiv-Physics",    ["science"]),
    ("http://export.arxiv.org/rss/q-bio",                     "arXiv-Biology",    ["science", "health"]),
    ("https://www.nature.com/nature.rss",                     "Nature",           ["science"]),
]


@dataclass
class RSSFeed:
    feed_id:       str                 # stable id, derived from name
    url:           str
    name:          str
    tags:          List[str] = field(default_factory=list)
    active:        bool      = True
    success_count: int       = 0
    fail_count:    int       = 0
    last_used:     float     = 0.0
    added_at:      float     = field(default_factory=time.time)

    @property
    def success_rate(self) -> float:
        total = self.success_count + self.fail_count
        if total == 0:
            return 0.5   # neutral prior — no track record yet, not "bad"
        return round(self.success_count / total, 3)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["success_rate"] = self.success_rate
        return d


class RSSFeedRegistry:
    """
    Thread-safe, persisted registry of configurable RSS feeds. One
    instance shared by search_providers.py (reads active feeds to query)
    and the management page (reads/writes everything, including inactive
    feeds and tags).
    """

    def __init__(self, path: str = DEFAULT_PATH):
        self._path = Path(path)
        self._lock = threading.RLock()
        self._feeds: Dict[str, RSSFeed] = {}
        self._load()

    # ── Query-time API (used by search_providers.py) ───────────────────────

    def active_feeds(self, tag_hint: Optional[str] = None) -> List[RSSFeed]:
        """
        Return active feeds, ranked by (tag match, success_rate) so the
        RSS tier tries the most relevant, most historically-successful
        feeds first. `tag_hint` is a loose domain guess (e.g. "science",
        "news", "tech") derived from the query — feeds with a matching
        tag are preferred but non-matching feeds are still included
        (better to try a general feed than return nothing).
        """
        with self._lock:
            feeds = [f for f in self._feeds.values() if f.active]

        def _rank(f: RSSFeed):
            tag_match = 1 if (tag_hint and tag_hint.lower() in [t.lower() for t in f.tags]) else 0
            return (tag_match, f.success_rate)

        feeds.sort(key=_rank, reverse=True)
        return feeds

    def mark_used(self, feed_id: str) -> None:
        with self._lock:
            f = self._feeds.get(feed_id)
            if f:
                f.last_used = time.time()

    # ── Outcome feedback (used by controller.py) ────────────────────────────

    def record_session_outcome(self, sources_used: List[str], confidence_score: float) -> None:
        """
        sources_used: list of WebPage.source strings from a finished
        session's pages, e.g. ["rss_BBC", "rss_NPR", "brave"]. Only
        "rss_<name>" entries are attributed here — other providers
        (brave/claude/stealth) aren't tracked by this registry.

        A session counts as a "success" for a contributing feed if the
        session's real confidence_score (from Evaluator.assess(), the
        existing LLM-based research-quality judgment) cleared 0.5 —
        reusing the quality signal that already exists rather than
        inventing a new one.
        """
        if not sources_used:
            return
        success = confidence_score >= 0.5
        with self._lock:
            touched = False
            for src in sources_used:
                if not src.startswith("rss_"):
                    continue
                name = src[len("rss_"):]
                feed = self._find_by_name(name)
                if feed is None:
                    continue
                if success:
                    feed.success_count += 1
                else:
                    feed.fail_count += 1
                feed.last_used = time.time()
                touched = True
            if touched:
                self._save()

    def _find_by_name(self, name: str) -> Optional[RSSFeed]:
        for f in self._feeds.values():
            if f.name == name:
                return f
        return None

    # ── Management API (used by the config page) ───────────────────────────

    def list_all(self) -> List[RSSFeed]:
        with self._lock:
            return sorted(self._feeds.values(), key=lambda f: f.name.lower())

    def add_feed(self, url: str, name: str, tags: Optional[List[str]] = None) -> RSSFeed:
        with self._lock:
            feed_id = self._slug(name)
            suffix = 1
            base_id = feed_id
            while feed_id in self._feeds:
                suffix += 1
                feed_id = f"{base_id}-{suffix}"
            feed = RSSFeed(feed_id=feed_id, url=url.strip(), name=name.strip(),
                            tags=[t.strip().lower() for t in (tags or []) if t.strip()])
            self._feeds[feed_id] = feed
            self._save()
            return feed

    def remove_feed(self, feed_id: str) -> bool:
        with self._lock:
            if feed_id in self._feeds:
                del self._feeds[feed_id]
                self._save()
                return True
            return False

    def set_active(self, feed_id: str, active: bool) -> bool:
        with self._lock:
            f = self._feeds.get(feed_id)
            if not f:
                return False
            f.active = active
            self._save()
            return True

    def set_tags(self, feed_id: str, tags: List[str]) -> bool:
        with self._lock:
            f = self._feeds.get(feed_id)
            if not f:
                return False
            f.tags = [t.strip().lower() for t in tags if t.strip()]
            self._save()
            return True

    def reset_stats(self, feed_id: str) -> bool:
        """Clear success/fail counts for a feed (e.g. after retagging it
        into a very different domain, where the old track record no
        longer means much)."""
        with self._lock:
            f = self._feeds.get(feed_id)
            if not f:
                return False
            f.success_count = 0
            f.fail_count = 0
            self._save()
            return True

    @staticmethod
    def _slug(name: str) -> str:
        return "".join(c.lower() if c.isalnum() else "-" for c in name).strip("-") or "feed"

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                for fd in data.get("feeds", []):
                    feed = RSSFeed(
                        feed_id=fd["feed_id"], url=fd["url"], name=fd["name"],
                        tags=fd.get("tags", []), active=fd.get("active", True),
                        success_count=fd.get("success_count", 0),
                        fail_count=fd.get("fail_count", 0),
                        last_used=fd.get("last_used", 0.0),
                        added_at=fd.get("added_at", time.time()),
                    )
                    self._feeds[feed.feed_id] = feed
                logger.info(f"[RSSFeedRegistry] Loaded {len(self._feeds)} feeds")
                return
        except Exception as e:
            logger.warning(f"[RSSFeedRegistry] Load failed, reseeding: {e}")

        # First run (or load failure) — seed with the original hardcoded list
        for url, name, tags in _SEED_FEEDS:
            feed_id = self._slug(name)
            self._feeds[feed_id] = RSSFeed(feed_id=feed_id, url=url, name=name, tags=tags)
        self._save()
        logger.info(f"[RSSFeedRegistry] Seeded {len(self._feeds)} default feeds")

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = {"feeds": [asdict(f) for f in self._feeds.values()]}
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"[RSSFeedRegistry] Save failed: {e}")


# ── Module-level singleton — one registry shared across the app ────────────
_registry_singleton: Optional[RSSFeedRegistry] = None
_singleton_lock = threading.Lock()


def get_rss_registry() -> RSSFeedRegistry:
    global _registry_singleton
    if _registry_singleton is None:
        with _singleton_lock:
            if _registry_singleton is None:
                _registry_singleton = RSSFeedRegistry()
    return _registry_singleton
