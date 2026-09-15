"""
pages/__init__.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Route registration hub for Lumina's NiceGUI application.

NiceGUI's @ui.page() decorator registers routes at *import time*.
Importing this package is all app.py needs to do — every route below
is registered automatically as a side-effect of the import chain.

Dependency order matters:
  shared      → must load first (CSS + utilities that pages depend on)
  page files  → each registers its own @ui.page route on import

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Adding a new page in the future — 2 steps only, app.py never touched:

  1. Create  pages/my_page.py  with @ui.page('/my-route')
  2. Add     from . import my_page   to this file

That's it. The route is live on next restart.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

# ── Shared utilities — must import before any page module ──────────────────────
from . import shared                  # noqa: F401  GLOBAL_CSS · _safe_ui · _drain_ui_queue

# ── Page route registrations ───────────────────────────────────────────────────
from . import main_page               # noqa: F401  GET /
from . import vision_page             # noqa: F401  GET /vision
from . import research_page           # noqa: F401  GET /research
from . import llm_page                # noqa: F401  GET /llm
from . import settings_page           # noqa: F401  GET /settings
from . import lumina_page             # noqa: F401  GET /lumina
from . import orchestrator_page       # noqa: F401  GET /orchestrator
from . import cognitive_dashboard_page # noqa: F401  GET /cognitive-dashboard
from . import rss_management_page     # noqa: F401  GET /rss-feeds

__all__ = [
    "shared",
    "main_page",
    "vision_page",
    "research_page",
    "llm_page",
    "settings_page",
    "lumina_page",
    "orchestrator_page",
    "cognitive_dashboard_page",
    "rss_management_page",
]
