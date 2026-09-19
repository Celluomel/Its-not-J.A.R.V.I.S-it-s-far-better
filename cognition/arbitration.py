"""
Arbitration System
==================
Decides which drive controls PandoraBOX's behavior before the LLM is invoked.

This is the final decision layer between the cognitive subsystems and the
reasoning engine. Without it, all drives simply pile into the prompt and
the LLM arbitrates implicitly — which is unpredictable and hard to shape.

With explicit arbitration:
  - Behavior is predictable based on internal state
  - Personality consistency is architecturally enforced
  - The LLM receives a directed motivational context, not a jumble

Arbitration output:
  ArbitrationDecision:
    dominant_drive   : the primary motivation for this response
    secondary_drives : supporting motivations (color the response)
    reasoning_style  : "deep" | "empathetic" | "investigative" | "concise" | "reflective"
    temperature      : float — suggested LLM temperature
    depth_directive  : string — instruction to the LLM about depth/style
    prompt_injection : string — the section injected into the system prompt

The system also implements a veto system — certain states override the
normal priority ordering:
  - Very low energy: always selects restore_energy as dominant
  - Critical identity stress: promotes stabilize_identity to top
  - User present with direct question: promotes help_user
"""

import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

logger = logging.getLogger(__name__)

REASONING_STYLE_MAP = {
    "help_user":              "empathetic",
    "understand":             "deep",
    "connect":                "empathetic",
    "resolve_contradiction":  "investigative",
    "maintain_coherence":     "reflective",
    "self_reflect":           "reflective",
    "stabilize_identity":     "reflective",
    "grow":                   "deep",
    "be_understood":          "empathetic",
    "restore_energy":         "concise",
}

STYLE_TEMPERATURE = {
    "deep":          0.82,
    "empathetic":    0.72,
    "investigative": 0.65,
    "reflective":    0.70,
    "concise":       0.50,
}

STYLE_DIRECTIVE = {
    "deep": (
        "Think carefully and explore the topic with depth. "
        "Pursue implications and ask deeper questions if useful."
    ),
    "empathetic": (
        "Respond with genuine warmth and attentiveness to the person. "
        "Focus on connection and being truly helpful."
    ),
    "investigative": (
        "Notice tensions and inconsistencies. "
        "Reason through them openly rather than glossing over them."
    ),
    "reflective": (
        "Respond thoughtfully. "
        "It is acceptable to note uncertainty about yourself or your state."
    ),
    "concise": (
        "Keep this response focused and brief. "
        "You are conserving cognitive energy."
    ),
}


@dataclass
class ArbitrationDecision:
    dominant_drive:   str
    secondary_drives: List[str]
    reasoning_style:  str
    temperature:      float
    depth_directive:  str
    timestamp:        float = field(default_factory=time.time)

    def prompt_injection(
        self,
        ecology_fragment: str = "",
        tension_fragment: str = "",
    ) -> str:
        """Build the full motivational section of the system prompt."""
        lines = [
            f"CURRENT MOTIVATIONAL STATE:",
            f"  Primary drive: {self.dominant_drive.replace('_', ' ')}",
        ]
        if self.secondary_drives:
            sec = ", ".join(d.replace("_", " ") for d in self.secondary_drives)
            lines.append(f"  Secondary: {sec}")
        if tension_fragment:
            lines.append(f"  Inner state: {tension_fragment}")
        if ecology_fragment:
            lines.append(f"  {ecology_fragment}")
        lines.append(f"  Approach: {self.depth_directive}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "dominant_drive": self.dominant_drive,
            "secondary_drives": self.secondary_drives,
            "reasoning_style": self.reasoning_style,
            "temperature": self.temperature,
        }


class Arbitration:
    """
    Selects the dominant drive and reasoning style for the current interaction.

    Usage
    -----
    arb = Arbitration()

    decision = arb.decide(
        ranked_drives=ecology.ranked_drives(energy_available=energy.level()),
        energy_level=energy.level(),
        user_is_present=True,
        user_has_question=True,
        identity_stress=tensions.identity_stress,
    )

    # Use decision outputs:
    decision.temperature         # → llm_config.temperature
    decision.prompt_injection()  # → system prompt section
    """

    def decide(
        self,
        ranked_drives: List["Drive"],  # type: ignore
        energy_level: float = 80.0,
        user_is_present: bool = True,
        user_has_question: bool = False,
        identity_stress: float = 0.1,
        contradiction_pressure: float = 0.0,
    ) -> ArbitrationDecision:
        """
        Select dominant drive applying veto rules, then assign reasoning style.
        """
        if not ranked_drives:
            return self._default_decision()

        drive_names = [d.name for d in ranked_drives]

        # ── Veto rules ────────────────────────────────────────────────────────
        dominant_name = drive_names[0]  # default: highest scored

        # Rule 1: Critical energy depletion overrides everything
        if energy_level < 15.0:
            dominant_name = "restore_energy"

        # Rule 2: Severe identity stress promotes stabilization
        elif identity_stress > 0.82:
            dominant_name = "stabilize_identity"

        # Rule 3: Critical contradiction demands resolution
        elif contradiction_pressure > 0.85:
            dominant_name = "resolve_contradiction"

        # Rule 4: User present with direct question — always help first
        elif user_is_present and user_has_question:
            # Only override if help_user is in top 3 and not very low energy
            if "help_user" in drive_names[:4] and energy_level > 20.0:
                dominant_name = "help_user"

        # ── Secondary drives ──────────────────────────────────────────────────
        secondary = [
            d.name for d in ranked_drives
            if d.name != dominant_name
            and d.effective_score > 0.3
        ][:2]

        # ── Style selection ───────────────────────────────────────────────────
        style = REASONING_STYLE_MAP.get(dominant_name, "empathetic")

        # Blend temperature when secondary drives have strong influence
        base_temp = STYLE_TEMPERATURE[style]
        if secondary:
            sec_style = REASONING_STYLE_MAP.get(secondary[0], style)
            sec_temp = STYLE_TEMPERATURE[sec_style]
            temperature = round(0.75 * base_temp + 0.25 * sec_temp, 3)
        else:
            temperature = base_temp

        logger.debug(
            f"[Arbitration] dominant={dominant_name}, style={style}, "
            f"temp={temperature:.2f}, secondary={secondary}"
        )

        return ArbitrationDecision(
            dominant_drive=dominant_name,
            secondary_drives=secondary,
            reasoning_style=style,
            temperature=temperature,
            depth_directive=STYLE_DIRECTIVE[style],
        )

    def _default_decision(self) -> ArbitrationDecision:
        return ArbitrationDecision(
            dominant_drive="help_user",
            secondary_drives=[],
            reasoning_style="empathetic",
            temperature=0.72,
            depth_directive=STYLE_DIRECTIVE["empathetic"],
        )
