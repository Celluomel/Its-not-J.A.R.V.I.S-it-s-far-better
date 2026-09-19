"""
Percept — Phase 6.0 (Universal Connector, minimal slice)
==========================================================
Not the full 6-phase vision from the architecture proposal (Universal
Connector with adapters/, registry.py, routing.py, Flux-as-peer-cognition,
actuators, temperature/proximity sensors). None of that has any existing
code to audit against — building it now would be speculation, not the
evidence-based methodology this project uses everywhere else.

This is the smallest real slice: the one thing this proposal concretely
diagnosed as broken today — cognition/presence_engine.py calls
self._llm.generate_bare() directly, bypassing respond() entirely, so every
camera-triggered utterance is cognitively invisible (never predicted
against, never remembered, never touches self-model). That's the same
bypass bug class already found and fixed for PandoraBOX/Flux dialogue (v66)
and outreach (v77) — this closes it for the presence/camera channel using
the shape the proposal recommends, so the abstraction is ready for a real
second sensor later without having to guess what that sensor needs.

Percept preserves what the proposal specifically called out as the part a
plain pipe would lose: provenance, timestamp, confidence, novelty,
salience — "the camera saw Fred" vs "the camera saw Fred at 98% confidence
300ms ago, right after Fred spoke, consistent with the prior prediction."
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class Percept:
    modality: str                       # e.g. "vision_presence", "audio_speech"
    source: str                         # e.g. "camera", "microphone_01"
    payload: Any                        # modality-specific content
    confidence: float = 1.0             # 0-1, how sure the source is
    salience: float = 0.5               # 0-1, how much this deserves attention
    novelty: float = 0.0                # 0-1, how unexpected (SurpriseFilter etc.)
    timestamp: float = field(default_factory=time.time)
    provenance: Dict[str, Any] = field(default_factory=dict)  # free-form context
    hop_count: int = 0                  # anti-echo guard for peer-sourced percepts
                                         # (Flux etc.) — see UniversalConnector's cap.
                                         # Not yet load-bearing: nothing currently
                                         # re-broadcasts a percept onward, so no path
                                         # can increment this past 0 today. Present
                                         # now so a future bidirectional exchange
                                         # (PandoraBOX workspace -> Flux -> back) has the
                                         # field ready rather than needing every
                                         # caller updated later.

    def age_seconds(self) -> float:
        return time.time() - self.timestamp
