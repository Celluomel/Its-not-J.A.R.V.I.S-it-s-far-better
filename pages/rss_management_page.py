"""
pages/rss_management_page.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RSS Feed Management — GET /rss-feeds

Configuration panel for the RSS feeds available to PandoraBOX's web-search
research pipeline (cognition/research_mcp). Each feed:

  - can be activated/deactivated via checkbox — inactive feeds are never
    queried by the RSS search tier
  - has editable domain tags (e.g. "science", "news", "tech") — when a
    query looks like it belongs to a domain, feeds tagged for that domain
    are tried first
  - shows a real success rate, tracked from actual research session
    outcomes (Evaluator.assess()'s confidence_score — the same quality
    signal the research pipeline already computes for itself, not a new
    metric invented for this page)

Backed by cognition/research_mcp/rss_registry.py.
"""

import logging
from typing import Any, Dict

from nicegui import ui

from pages.shared import GLOBAL_CSS
from cognition.research_mcp.rss_registry import get_rss_registry

logger = logging.getLogger(__name__)


# ── Shared visual helpers (matches cognitive_dashboard_page.py's style) ────

def _badge(text: str, color: str = 'gray') -> None:
    color_map = {
        'green':  ('bg-emerald-900', 'text-emerald-300', 'border-emerald-700'),
        'yellow': ('bg-yellow-900',  'text-yellow-300',  'border-yellow-700'),
        'red':    ('bg-red-900',     'text-red-300',     'border-red-700'),
        'blue':   ('bg-blue-900',    'text-blue-300',    'border-blue-700'),
        'purple': ('bg-purple-900',  'text-purple-300',  'border-purple-700'),
        'gray':   ('bg-slate-800',   'text-slate-300',   'border-slate-600'),
    }
    bg, fg, border = color_map.get(color, color_map['gray'])
    ui.label(text).classes(
        f'{bg} {fg} border {border} text-xs px-2 py-0.5 rounded-full font-mono'
    )


def _success_color(rate: float, total: int) -> str:
    if total == 0:
        return 'gray'
    if rate >= 0.65:
        return 'green'
    if rate >= 0.4:
        return 'yellow'
    return 'red'


@ui.page('/rss-feeds')
async def rss_management_page():
    ui.add_head_html(f'<style>{GLOBAL_CSS}</style>')
    registry = get_rss_registry()

    ui.label('RSS Feed Management').classes('text-2xl font-bold text-slate-200 mt-4 mb-1')
    ui.label(
        'Feeds used by the research web-search pipeline. Deactivated feeds are '
        'never queried. Tags route domain-relevant queries (e.g. a science '
        'question prefers feeds tagged "science") toward the right feeds first. '
        'Success rate reflects real research session outcomes, not a static guess.'
    ).classes('text-slate-500 text-sm mb-6 max-w-3xl')

    feed_list_container = ui.column().classes('w-full gap-3')

    def refresh():
        feed_list_container.clear()
        with feed_list_container:
            for feed in registry.list_all():
                _render_feed_card(feed, registry, refresh)

    def _render_feed_card(feed, registry, refresh_fn):
        total = feed.success_count + feed.fail_count
        rate = feed.success_rate
        color = _success_color(rate, total)

        with ui.element('div').classes(
            'bg-slate-900 border border-slate-700 rounded-xl p-4 w-full'
        ):
            with ui.row().classes('items-center justify-between w-full gap-3'):
                with ui.row().classes('items-center gap-3 flex-grow'):
                    active_cb = ui.checkbox(value=feed.active).props('dense')

                    def _on_active_change(e, fid=feed.feed_id):
                        registry.set_active(fid, e.value)
                        ui.notify(
                            f"{feed.name} {'activated' if e.value else 'deactivated'}",
                            type='positive' if e.value else 'warning',
                        )
                    active_cb.on_value_change(_on_active_change)

                    with ui.column().classes('gap-0'):
                        ui.label(feed.name).classes('text-slate-200 font-semibold text-sm')
                        ui.label(feed.url).classes('text-slate-500 text-xs font-mono truncate max-w-md')

                with ui.row().classes('items-center gap-2'):
                    if total > 0:
                        _badge(f'{rate:.0%} ({feed.success_count}/{total})', color)
                    else:
                        _badge('no track record yet', 'gray')

                    def _do_reset(fid=feed.feed_id):
                        registry.reset_stats(fid)
                        ui.notify(f"Stats reset for {feed.name}", type='info')
                        refresh_fn()
                    ui.button(icon='restart_alt', on_click=_do_reset) \
                        .props('flat dense round size=sm') \
                        .classes('text-slate-500').tooltip('Reset success/fail stats')

                    def _do_delete(fid=feed.feed_id, fname=feed.name):
                        registry.remove_feed(fid)
                        ui.notify(f"Removed {fname}", type='warning')
                        refresh_fn()
                    ui.button(icon='delete', on_click=_do_delete) \
                        .props('flat dense round size=sm') \
                        .classes('text-red-400').tooltip('Remove feed')

            # Tags row — editable
            with ui.row().classes('items-center gap-2 mt-3 flex-wrap'):
                ui.label('tags:').classes('text-slate-500 text-xs')
                tags_input = ui.input(
                    value=', '.join(feed.tags),
                    placeholder='news, science, tech...',
                ).props('dense borderless').classes(
                    'text-slate-300 text-xs bg-slate-800 rounded px-2 flex-grow max-w-sm'
                )

                def _on_tags_blur(e, fid=feed.feed_id):
                    new_tags = [t.strip() for t in tags_input.value.split(',') if t.strip()]
                    registry.set_tags(fid, new_tags)
                    ui.notify(f"Tags updated: {', '.join(new_tags) or '(none)'}", type='positive')
                tags_input.on('blur', _on_tags_blur)

    refresh()

    # ── Add new feed ─────────────────────────────────────────────────────
    ui.separator().classes('my-6 border-slate-700')
    ui.label('Add a new feed').classes('text-slate-300 font-semibold text-sm uppercase tracking-wide mb-2')

    with ui.element('div').classes('bg-slate-900 border border-slate-700 rounded-xl p-4 w-full'):
        with ui.row().classes('w-full gap-3 items-end flex-wrap'):
            name_input = ui.input('Name', placeholder='e.g. arXiv CS.AI').classes('w-48')
            url_input  = ui.input('RSS URL', placeholder='http://export.arxiv.org/rss/cs.AI').classes('flex-grow min-w-96')
            tags_new_input = ui.input('Tags (comma-separated)', placeholder='science, tech').classes('w-64')

            def _do_add():
                name = name_input.value.strip()
                url  = url_input.value.strip()
                if not name or not url:
                    ui.notify('Name and URL are both required', type='negative')
                    return
                if not (url.startswith('http://') or url.startswith('https://')):
                    ui.notify('URL must start with http:// or https://', type='negative')
                    return
                tags = [t.strip() for t in tags_new_input.value.split(',') if t.strip()]
                registry.add_feed(url, name, tags)
                ui.notify(f'Added {name}', type='positive')
                name_input.value = ''
                url_input.value = ''
                tags_new_input.value = ''
                refresh()

            ui.button('Add Feed', icon='add', on_click=_do_add).props('color=primary')

    ui.label(
        'Example science feeds: arXiv (http://export.arxiv.org/rss/cs.AI, '
        'http://export.arxiv.org/rss/physics), Nature News '
        '(https://www.nature.com/nature.rss)'
    ).classes('text-slate-600 text-xs mt-2 max-w-3xl')
