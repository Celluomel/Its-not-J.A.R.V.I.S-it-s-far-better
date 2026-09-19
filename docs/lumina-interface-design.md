# PandoraBOX interface design v1

Status: proposed design specification; not an implemented interface or Figma file.

## Direction

Conversation is the primary workspace. A vivid, unframed organism shares the
canvas with readable chat. A cognitive inspector is collapsed by default and
opens beside chat without navigating away or resetting the conversation.

Use near-black neutral surfaces (#111315), pale text (#F0F3F4), muted secondary
text (#A9B2B7), and restrained borders (#343A40). Use mint (#65DFC0) for connected
state, coral (#F18C94) for attention, and gold (#EAC76C) for uncertainty. The
organism combines pearlescent membranes with mint and coral filaments. Avoid
decorative background orbs. Typography: system sans, 14-16px body, 24px page
title, zero letter spacing. Minimum icon hit area 44px; visible focus rings.

## Screen 1: Conversation

Desktop reference frame: 1440 x 960. Header: 56px. Navigation rail: 64px.
Remaining workspace: organism occupies approximately 34%, chat 66%. Constrain
message width to 680px. Anchor the composer below the transcript. Keep the
organism unframed and vertically centered without obscuring text or controls.

Header: PandoraBOX, connection state, cognitive-inspector icon, settings icon.
Rail: new conversation, history, memory, voice. Tooltips name every icon.
Composer: attachment, multiline input, microphone, send; stop replaces send
while a turn is active. Preserve the draft when changing inspection views.
Show history in a temporary sidebar. Do not add feature-explanation copy.

## Screen 2: Cognitive inspector open

Use the same conversation and organism state as screen 1. Right inspector:
400px initial width, resizable from 340px to 520px. Chat must retain at least
480px on desktop; reduce organism space first. Below 1100px, use an overlay
inspector; below 720px, use a full-screen inspector with a back control.

Tabs: Overview, Memory, Relations, Activity, Diagnostics. Use compact rows,
dividers, and disclosure groups rather than nested cards. Overview shows
runtime state, current activity, last update, and available grounded metrics.
Memory shows retrieved records and provenance. Relations shows typed edges
with source, confidence, and validity. Activity shows recorded events, not
an invented inner monologue. Diagnostics includes STT, generation, and TTS
timings. Missing signals read Unavailable; stale signals retain a timestamp.

Selecting an event opens its evidence within the inspector. Closing it returns
focus to its trigger and preserves transcript position. Opening the inspector
must not initialize cognitive engines or cause any runtime mutation.

## Screen 3: Immersive voice

Retain the header and conversation identity. Expand the organism across the
available canvas. Show a compact live transcript beneath it and a bottom bar
with microphone, interruption, text-view, and end-voice controls. Listening,
processing, speaking, muted, and disconnected states use text and motion,
never color alone. Exiting voice retains the conversation and draft.

## Organism motion specification

- Idle: slow membrane deformation; explicitly aesthetic baseline motion.
- Listening: localized movement driven by microphone amplitude only when enabled.
- Processing: directional filament flow driven by recorded processing state.
- Retrieval: brief internal paths triggered by actual retrieval events.
- Speaking: surface motion driven by playback amplitude, not generated tokens.
- Interrupted: speaking motion stops with audio cancellation; return to listening.
- Disconnected: settle into a quiet static state with a visible status label.

Render with Three.js and React Three Fiber. Interpolate telemetry locally;
never use React state updates for each animation frame. Target 60 FPS on the
reference desktop, adapt to 30 FPS on constrained hardware, pause when hidden,
and provide reduced-motion and graphics-unavailable alternatives. Do not
present aesthetic motion or self-reported states as proof of consciousness.

## Figma structure and prototype connections

Pages: Foundations, Components, Desktop, Mobile, Motion Reference.
Foundations: semantic color variables, spacing, typography, elevation, focus.
Components: icon button, composer, message, status, event row, inspector tabs.
Variants: default, hover, focused, disabled, loading, error where applicable.
Desktop frames: Conversation, Inspector Open, Voice. Mobile frames: 390 x 844
equivalents with compact organism and full-screen inspector.

Connections: inspector trigger opens Inspector Open; close restores Conversation;
voice trigger opens Voice; text-view restores Conversation. Prototype tab
selection, history selection, event disclosure, permission failure, disconnect,
and reduced-motion state. All displayed sample telemetry must be marked as demo
data in the design file; implementation must use actual backend events.

## Implementation handoff

First extract UI-independent conversation and telemetry services from
pages/main_page.py and pages/cognitive_dashboard_page.py. Keep one authoritative
runtime. Build typed HTTP and WebSocket contracts with turn IDs, sequence
numbers, cancellation, bounded queues, and reconnect snapshots. Then implement
the three views, followed by audio transport and remaining page migration.

Validate keyboard navigation, transcript scrolling during streaming, panel
resizing, reconnect recovery, interruption, mobile text fitting, and graphics
fallback. Measure voice latency separately from rendering performance.
