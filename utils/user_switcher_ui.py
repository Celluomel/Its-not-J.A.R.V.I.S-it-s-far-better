"""
utils/user_switcher_ui.py

Builds the user switcher widget embedded in the Robot Agent chat page.
Designed to sit in the chat header row — compact by default,
expands into a management dialog.

Public API
----------
build_user_chip(user_manager) → element
    Small chip showing the active user's name + colour dot.
    Click → opens the user picker/manager dialog.

open_user_dialog(user_manager, vision_state) → coroutine
    Full user management dialog: list, create, edit, delete, link face.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Optional

from nicegui import ui

if TYPE_CHECKING:
    from managers.user_manager import UserManager, UserProfile
    from managers.vision_manager import StreamingVisionManager

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
#  Active-user chip (inline, always visible)
# ─────────────────────────────────────────────────────────────────────────────

def build_user_chip(user_manager, on_change=None, vision=None):
    """
    Returns a clickable chip element showing the active user.
    Call `refresh_chip(chip_refs)` after switching users to update it.

    on_change : optional async callback() called when active user changes
    vision    : optional StreamingVisionManager — forwarded to the edit
                dialog so face IDs can be linked directly from the UI
    """
    refs = {}

    with ui.element("div").classes("user-chip cursor-pointer select-none") as chip:
        with ui.row().classes("items-center gap-1.5 px-3 py-1 rounded-full glass-card"):
            refs["dot"] = ui.element("div").style(
                f"width:8px;height:8px;border-radius:50%;"
                f"background:{user_manager.active.color};flex-shrink:0"
            )
            refs["name"] = ui.label(user_manager.active.display_name).classes(
                "text-xs font-medium text-slate-300"
            )
            ui.icon("expand_more", size="xs").classes("text-slate-500")

    async def _click():
        await open_user_dialog(user_manager, refs, on_change, vision=vision)

    chip.on("click", _click)
    return chip, refs


def refresh_chip(refs: dict, user_manager):
    """Update the chip after an active-user change."""
    try:
        profile = user_manager.active
        refs["dot"].style(
            f"width:8px;height:8px;border-radius:50%;"
            f"background:{profile.color};flex-shrink:0"
        )
        refs["name"].set_text(profile.display_name)
    except Exception as e:
        logger.debug(f"refresh_chip: {e}")


# ─────────────────────────────────────────────────────────────────────────────
#  User management dialog
# ─────────────────────────────────────────────────────────────────────────────

async def open_user_dialog(user_manager, chip_refs: dict = None, on_change=None, vision=None):
    """Full user management dialog."""

    with ui.dialog().props("persistent") as dialog, \
         ui.card().classes("w-full max-w-md p-0 bg-slate-900 rounded-2xl overflow-hidden"):

        # ── Header ────────────────────────────────────────────────────────
        with ui.row().classes(
            "w-full items-center justify-between px-5 py-4 "
            "border-b border-slate-700"
        ):
            with ui.row().classes("items-center gap-2"):
                ui.icon("group").classes("text-purple-400")
                ui.label("Users").classes("text-base font-bold text-slate-100")
            ui.button(icon="close", on_click=dialog.close).props("flat round dense").classes(
                "text-slate-500"
            )

        # ── User list ─────────────────────────────────────────────────────
        user_list = ui.column().classes("w-full gap-1 px-4 py-3 max-h-64 overflow-y-auto")

        def _render_users():
            user_list.clear()
            with user_list:
                users = user_manager.named_users()
                if not users:
                    ui.label("No users yet — create one below.").classes(
                        "text-xs text-slate-500 py-2 text-center w-full"
                    )
                    return

                for profile in users:
                    is_active = profile.id == user_manager.active_id
                    row_cls = (
                        "w-full items-center gap-3 p-2 rounded-xl cursor-pointer "
                        + ("bg-slate-700" if is_active else "hover:bg-slate-800")
                    )
                    with ui.row().classes(row_cls) as row:
                        # Colour dot
                        ui.element("div").style(
                            f"width:28px;height:28px;border-radius:50%;"
                            f"background:{profile.color};flex-shrink:0;"
                            f"display:flex;align-items:center;justify-content:center;"
                            f"font-size:13px;color:white;font-weight:600"
                        ).on(
                            "click",
                            lambda p=profile: _switch(p)
                        ).tooltip(profile.display_name[0].upper())
                        # Actually put the initial inside the dot
                        # (NiceGUI doesn't support children inside element easily,
                        #  so we use a label positioned absolutely via html)

                        # Name + meta
                        with ui.column().classes("flex-1 gap-0").on(
                            "click", lambda p=profile: _switch(p)
                        ):
                            with ui.row().classes("items-center gap-2"):
                                ui.label(profile.display_name).classes(
                                    "text-sm font-medium text-slate-200"
                                )
                                if is_active:
                                    ui.html(
                                        '<span style="background:#6366f1;color:white;'
                                        'border-radius:20px;padding:1px 8px;font-size:0.65rem;'
                                        'font-weight:600">active</span>'
                                    )
                            # Face count + notes preview
                            meta_parts = []
                            if profile.face_ids:
                                meta_parts.append(f"📷 {len(profile.face_ids)} face{'s' if len(profile.face_ids) > 1 else ''}")
                            if profile.notes:
                                meta_parts.append(profile.notes[:40] + ("…" if len(profile.notes) > 40 else ""))
                            if meta_parts:
                                ui.label(" · ".join(meta_parts)).classes(
                                    "text-xs text-slate-500"
                                )

                        # Edit / Delete buttons
                        with ui.row().classes("gap-1 flex-shrink-0"):
                            ui.button(icon="edit", on_click=lambda p=profile: _edit(p)).props(
                                "flat round dense size=sm"
                            ).classes("text-slate-400")
                            ui.button(icon="delete", on_click=lambda p=profile: _delete(p)).props(
                                "flat round dense size=sm"
                            ).classes("text-red-500")

        def _switch(profile):
            user_manager.set_active(profile.id)
            if chip_refs:
                refresh_chip(chip_refs, user_manager)
            if on_change:
                asyncio.create_task(on_change()) if asyncio.get_event_loop().is_running() else None
            _render_users()
            ui.notify(f"👤 Switched to {profile.display_name}", type="info",
                      position="bottom-right", timeout=2000)

        async def _delete(profile):
            with ui.dialog() as confirm_d, ui.card().classes(
                "p-4 bg-slate-800 rounded-xl"
            ):
                ui.label(f'Delete "{profile.display_name}"?').classes(
                    "text-sm font-semibold text-slate-200 mb-3"
                )
                ui.label("Their conversation history remains in memory.").classes(
                    "text-xs text-slate-400 mb-4"
                )
                with ui.row().classes("gap-2 justify-end"):
                    ui.button("Cancel", on_click=confirm_d.close).props("flat").classes(
                        "text-slate-400"
                    )
                    async def _do_delete():
                        user_manager.delete(profile.id)
                        if chip_refs:
                            refresh_chip(chip_refs, user_manager)
                        confirm_d.close()
                        _render_users()
                        ui.notify(f"Deleted {profile.display_name}", type="warning",
                                  position="bottom-right", timeout=2000)
                    ui.button("Delete", on_click=_do_delete).props("color=red")
            confirm_d.open()

        async def _edit(profile):
            await _open_edit_dialog(profile, user_manager, vision, chip_refs, on_change,
                                    refresh_callback=_render_users)

        _render_users()

        ui.separator().classes("opacity-20 mx-4")

        # ── Create new user ───────────────────────────────────────────────
        with ui.column().classes("w-full px-4 pb-4 gap-3 mt-3"):
            ui.label("New User").classes("text-xs font-semibold text-slate-400 uppercase")
            with ui.row().classes("w-full items-center gap-2"):
                name_input = ui.input(placeholder="Display name…").props(
                    "outlined dense borderless"
                ).classes("flex-1")

                async def _create():
                    name = name_input.value.strip()
                    if not name:
                        ui.notify("Enter a name first", type="warning", position="bottom-right")
                        return
                    profile = user_manager.create(name)
                    user_manager.set_active(profile.id)
                    name_input.value = ""
                    if chip_refs:
                        refresh_chip(chip_refs, user_manager)
                    if on_change:
                        asyncio.create_task(on_change())
                    _render_users()
                    ui.notify(f"✅ Created {profile.display_name}", type="positive",
                              position="bottom-right", timeout=2000)

                ui.button(icon="add", on_click=_create).props(
                    "round dense color=indigo"
                ).classes("flex-shrink-0")

    dialog.open()


# ─────────────────────────────────────────────────────────────────────────────
#  Edit dialog (name, notes, colour, face links)
# ─────────────────────────────────────────────────────────────────────────────

async def _open_edit_dialog(profile, user_manager, vision, chip_refs, on_change, refresh_callback):
    with ui.dialog().props("persistent") as edit_d, \
         ui.card().classes("w-full max-w-sm p-0 bg-slate-900 rounded-2xl overflow-hidden"):

        with ui.row().classes(
            "w-full items-center justify-between px-5 py-4 border-b border-slate-700"
        ):
            ui.label(f"Edit — {profile.display_name}").classes(
                "text-sm font-bold text-slate-200"
            )
            ui.button(icon="close", on_click=edit_d.close).props("flat round dense").classes(
                "text-slate-500"
            )

        with ui.column().classes("w-full px-5 py-4 gap-4"):

            # Name
            name_inp = ui.input("Display name", value=profile.display_name).props(
                "outlined dense"
            ).classes("w-full dark-input")

            # Notes
            notes_inp = ui.textarea("Notes / bio", value=profile.notes).props(
                "outlined rows=3"
            ).classes("w-full dark-input").tooltip(
                "Injected into the system prompt so the AI knows this person"
            )

            # Colour picker
            with ui.row().classes("items-center gap-3"):
                ui.label("Accent colour").classes("text-xs text-slate-400 w-24")
                from managers.user_manager import _PALETTE
                for hex_color in _PALETTE:
                    is_sel = profile.color == hex_color
                    dot = ui.element("div").style(
                        f"width:20px;height:20px;border-radius:50%;"
                        f"background:{hex_color};cursor:pointer;"
                        + (f"outline:2px solid white;outline-offset:2px" if is_sel else "")
                    )
                    dot.on("click", lambda c=hex_color: _set_color(c))

            selected_color = {"value": profile.color}

            def _set_color(c):
                selected_color["value"] = c

            # Linked faces
            with ui.expansion("📷 Linked Faces", icon="face").classes("w-full"):
                face_col = ui.column().classes("gap-2 w-full py-2")

                def _render_faces():
                    face_col.clear()
                    with face_col:
                        if not profile.face_ids:
                            ui.label("No faces linked").classes("text-xs text-slate-500")
                        for fid in profile.face_ids:
                            with ui.row().classes("items-center justify-between w-full"):
                                ui.label(fid).classes("text-xs text-slate-300 font-mono")
                                async def _unlink(f=fid):
                                    user_manager.unlink_face(profile.id, f)
                                    _render_faces()
                                ui.button(icon="link_off", on_click=_unlink).props(
                                    "flat round dense size=xs"
                                ).classes("text-red-400")

                _render_faces()

                # Link a face from the current detected set
                if vision:
                    ui.label("Link detected face:").classes("text-xs text-slate-500 mt-2")
                    with ui.row().classes("gap-2 flex-wrap mt-1"):
                        all_face_ids = list(getattr(vision, "face_encodings", {}).keys())
                        linked = set(profile.face_ids)
                        unlinked = [f for f in all_face_ids if f not in linked]
                        if unlinked:
                            for fid in unlinked:
                                fname = getattr(vision, "face_names", {}).get(fid, fid)
                                async def _link(f=fid):
                                    user_manager.link_face(profile.id, f)
                                    _render_faces()
                                ui.button(
                                    f"+ {fname}",
                                    on_click=_link
                                ).props("flat dense outline color=blue").classes("text-xs")
                        else:
                            ui.label("No unlinked faces available").classes(
                                "text-xs text-slate-600"
                            )

        # Save
        with ui.row().classes("w-full justify-end gap-2 px-5 pb-4"):
            ui.button("Cancel", on_click=edit_d.close).props("flat").classes("text-slate-400")

            async def _save():
                user_manager.update(
                    profile.id,
                    display_name=name_inp.value.strip() or profile.display_name,
                    notes=notes_inp.value.strip(),
                    color=selected_color["value"],
                )
                if chip_refs:
                    refresh_chip(chip_refs, user_manager)
                if on_change:
                    asyncio.create_task(on_change())
                refresh_callback()
                edit_d.close()
                ui.notify("✅ Saved", type="positive", position="bottom-right", timeout=2000)

            ui.button("Save", on_click=_save).props("color=indigo")

    edit_d.open()
