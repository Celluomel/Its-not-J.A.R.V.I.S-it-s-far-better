# PandoraBOX interface: first migration milestone

React/TypeScript frontend with a Three.js organism, streaming conversation,
read-only cognitive inspector, hands-free voice conversation through local VAD
and PandoraBOX STT, and playback through PandoraBOX TTS. No second cognitive runtime is
started.

## Run

Start PandoraBOX normally using its working Python environment. Restart an existing
instance to register the new `/api/interface` routes. Then, in `frontend`:

```powershell
npm install
npm run dev -- --port 5173
```

Open http://127.0.0.1:5173/next/. The development proxy targets port 8080 by
default. Set `LUMINA_BACKEND_URL` before starting Vite for another backend port.

For the existing launcher, run `npm run build`, then restart PandoraBOX. The built
interface is served at `/next/` on the same port as NiceGUI. The old UI remains
at `/`. The new interface deliberately does not replace the default launcher.

## Scope and limits

- The browser stores the last 200 transcript messages locally. This is a single
  ongoing conversation with the active PandoraBOX user; it is not isolated sessions.
  Existing cognitive memory continues through PersonaBridge. NiceGUI transcript
  history is not imported. Do not send simultaneous turns from both interfaces.
- Inspector reads existing runtime attributes and events recorded by this
  adapter. It does not call the old collector that initializes lazy engines.
  Memory/relationship inspection and full page migration remain future work.
- Closing a streaming request stops frontend output. An in-flight threaded
  inference call may finish before the worker releases the turn lock. Backend
  lifecycle effects already performed cannot be undone by interrupting display.
- Voice mode requests microphone permission once, keeps the local VAD listening,
  submits each detected utterance to PandoraBOX automatically, and pauses detection
  while STT or TTS is active. It plays completed responses. Sentence streaming,
  playback-amplitude animation, and simultaneous barge-in remain future work.
- Idle organism movement is aesthetic. Processing reflects runtime state;
  listening energy comes from the microphone. No fabricated cognitive scores.
- Settings links currently use the default local NiceGUI port 8080.

## Verification

`npm run build` typechecks and builds. `npm test` checks streaming boundaries.
The standalone API contract tests in `tests/test_interface_api.py` use a fake
runtime and must not boot cognitive engines or call real inference providers.
