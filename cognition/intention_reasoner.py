"""Structured, inspectable reasoning for committed cognitive intentions.

This module records concise deliberation summaries, not hidden token-level
model reasoning. It compares executable hypotheses against Lumina's learned
world/self dynamics and preserves the evidence, uncertainty and decision that
the planner actually used.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import RLock
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

SAVE_PATH = "data/persona/reasoning_episodes.json"
MAX_EPISODES = 200

OPERATION_STYLE = {
    "memory_recall": "analytical",
    "self_question": "philosophical",
    "bisociative_hypothesis": "creative",
    "web_search": "technical",
    "user_question": "social",
}

# Conservative priors are used only when WSDM has insufficient comparable
# experience. Their low confidence makes that limitation explicit.
OPERATION_PRIORS = {
    "memory_recall": (0.02, 0.025, 0.38),
    "self_question": (0.05, 0.040, 0.34),
    "bisociative_hypothesis": (0.07, 0.050, 0.28),
    "web_search": (0.08, 0.065, 0.32),
    "user_question": (0.03, 0.060, 0.42),
}


@dataclass
class ReasoningEvidence:
    source: str
    claim: str
    confidence: float


@dataclass
class ReasoningHypothesis:
    operation: str
    statement: str
    expected_cost: float
    expected_gain: float
    confidence: float
    score: float
    evidence_source: str


@dataclass
class ReasoningEpisode:
    id: str
    intention_id: str
    question: str
    evidence: List[ReasoningEvidence]
    hypotheses: List[ReasoningHypothesis]
    selected_operation: str
    recommended_sequence: List[str]
    conclusion: str
    decision_rationale: str
    confidence: float
    uncertainties: List[str]
    status: str = "decided"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    outcomes: List[Dict[str, Any]] = field(default_factory=list)
    initial_confidence: float = 0.0


class IntentionReasoner:
    """Compare action hypotheses for one goal and learn from their outcomes."""

    def __init__(self, organism: Any, path: str = SAVE_PATH):
        self._organism = organism
        self._path = Path(path)
        self._lock = RLock()
        self._episodes: Dict[str, ReasoningEpisode] = {}
        self._load()

    def deliberate(self, goal: Any, operations: List[str]) -> ReasoningEpisode:
        goal_id = str(getattr(goal, "id", "unknown"))
        topic = str(getattr(goal, "topic", "an unresolved goal"))
        origin = str(getattr(goal, "origin", "unknown"))
        context = self._read_context()

        evidence = [
            ReasoningEvidence(
                source="goal_engine",
                claim=f"The active goal originates from {origin} and concerns: {topic}",
                confidence=float(getattr(goal, "priority", 0.5)),
            )
        ]
        if context:
            evidence.extend([
                ReasoningEvidence(
                    source="resource_economy",
                    claim=f"Current cognitive energy is {context[0]:.2f}",
                    confidence=1.0,
                ),
                ReasoningEvidence(
                    source="relational_state",
                    claim=f"Current user trust context is {context[5]:.2f}",
                    confidence=0.8,
                ),
            ])

        hypotheses = [self._evaluate(operation, context) for operation in operations]
        ranked = sorted(hypotheses, key=lambda item: item.score, reverse=True)

        # Recall is an evidential prerequisite, not merely another competing
        # tactic. Keep it first when available, then rank the true alternatives.
        sequence: List[str] = []
        if "memory_recall" in operations:
            sequence.append("memory_recall")
        sequence.extend(
            item.operation for item in ranked if item.operation not in sequence
        )

        selected = sequence[0] if sequence else operations[0]
        selected_hypothesis = next(
            item for item in hypotheses if item.operation == selected
        )
        mean_confidence = sum(item.confidence for item in hypotheses) / max(1, len(hypotheses))
        uncertainties: List[str] = []
        if any(item.evidence_source == "prior" for item in hypotheses):
            uncertainties.append("Some alternatives lack enough comparable WSDM outcomes")
        if len(ranked) > 1 and abs(ranked[0].score - ranked[1].score) < 0.08:
            uncertainties.append("The two strongest alternatives have similar expected utility")
        if mean_confidence < 0.5:
            uncertainties.append("The decision is provisional because predictive confidence is low")

        rationale = (
            f"Begin with {selected} as the evidential prerequisite; then prefer "
            f"{ranked[0].operation} among the remaining alternatives "
            f"(expected gain {ranked[0].expected_gain:.3f}, cost "
            f"{ranked[0].expected_cost:.3f}, confidence {ranked[0].confidence:.2f})."
        )
        episode = ReasoningEpisode(
            id=f"reason-{uuid.uuid4().hex[:12]}",
            intention_id=f"goal:{goal_id}",
            question=f"Which sequence is most likely to make grounded progress on: {topic}?",
            evidence=evidence,
            hypotheses=hypotheses,
            selected_operation=selected,
            recommended_sequence=sequence,
            conclusion=f"Use the ordered strategy: {' -> '.join(sequence)}",
            decision_rationale=rationale,
            confidence=round(mean_confidence, 3),
            uncertainties=uncertainties,
            initial_confidence=round(mean_confidence, 3),
        )
        with self._lock:
            self._episodes[episode.id] = episode
            self._prune()
            self._save_locked()
        logger.info(
            "[IntentionReasoner] %s | confidence=%.2f | %s",
            episode.intention_id, episode.confidence, episode.conclusion,
        )
        self._broadcast(
            "intention_reasoner",
            f"[IntentionDecision] goal='{topic[:60]}'; selected={selected}; "
            f"sequence={' -> '.join(sequence)}; confidence={episode.confidence:.2f}",
            0.58,
        )
        if uncertainties:
            self._broadcast(
                "intention_reasoner.uncertainty",
                f"[ReasoningUncertainty] goal='{topic[:60]}'; {uncertainties[0][:120]}",
                0.64,
            )
        return episode

    def record_outcome(
        self, episode_id: str, operation: str, success: bool, result: str
    ) -> Optional[ReasoningEpisode]:
        with self._lock:
            episode = self._episodes.get(episode_id)
            if episode is None:
                return None
            if episode.initial_confidence <= 0.0:
                episode.initial_confidence = episode.confidence
            episode.outcomes.append({
                "operation": operation,
                "success": bool(success),
                "result": str(result)[:240],
                "observed_at": time.time(),
            })
            episode.updated_at = time.time()
            episode.status = "validated" if success else "revision_needed"
            episode.evidence.append(ReasoningEvidence(
                source="verified_outcome",
                claim=(f"{operation} succeeded: " if success else f"{operation} failed: ")
                      + str(result)[:180],
                confidence=1.0,
            ))
            successes = sum(1 for item in episode.outcomes if item.get("success"))
            total = len(episode.outcomes)
            observed_reliability = (successes + 1.0) / (total + 2.0)
            episode.confidence = round(
                max(0.1, min(0.95,
                    episode.initial_confidence * 0.65 + observed_reliability * 0.35
                )),
                3,
            )
            self._save_locked()
        self._broadcast(
            "intention_reasoner.outcome",
            f"[ReasoningOutcome] intention={episode.intention_id}; "
            f"operation={operation}; result={'success' if success else 'failure'}",
            0.54 if success else 0.67,
        )
        return episode

    def update_decision(
        self,
        episode_id: str,
        rationale: str,
        uncertainties: List[str],
        status: str,
    ) -> Optional[ReasoningEpisode]:
        """Synchronize the recorded reasoning with the planner's operative decision."""
        with self._lock:
            episode = self._episodes.get(episode_id)
            if episode is None:
                return None
            episode.decision_rationale = str(rationale)[:500]
            episode.uncertainties = list(dict.fromkeys(uncertainties))[:6]
            episode.status = status
            episode.updated_at = time.time()
            self._save_locked()
            return episode

    def get(self, episode_id: str) -> Optional[ReasoningEpisode]:
        return self._episodes.get(episode_id)

    def latest_summary(self) -> str:
        if not self._episodes:
            return ""
        episode = max(self._episodes.values(), key=lambda item: item.updated_at)
        uncertainty = episode.uncertainties[0] if episode.uncertainties else "none material"
        return (
            f"[Reasoning] {episode.conclusion}. Confidence={episode.confidence:.2f}; "
            f"uncertainty: {uncertainty}."
        )

    def _broadcast(self, source: str, content: str, priority: float) -> None:
        """Publish a compact reasoning artifact to the shared workspace."""
        try:
            workspace = getattr(self._organism, "workspace", None)
            if workspace is not None and hasattr(workspace, "broadcast"):
                workspace.broadcast(
                    source=source,
                    content=content[:240],
                    priority=max(0.0, min(0.75, priority)),
                )
        except Exception as exc:
            logger.debug("[IntentionReasoner] workspace broadcast failed: %s", exc)

    def _read_context(self) -> Optional[List[float]]:
        try:
            loop = getattr(self._organism, "_loop", None)
            model = getattr(loop, "_world_self_dynamics", None) if loop else None
            return model.read_state() if model is not None else None
        except Exception as exc:
            logger.debug("[IntentionReasoner] context unavailable: %s", exc)
            return None

    def _evaluate(
        self, operation: str, context: Optional[List[float]]
    ) -> ReasoningHypothesis:
        cost, gain, confidence = OPERATION_PRIORS.get(operation, (0.06, 0.03, 0.25))
        source = "prior"
        try:
            loop = getattr(self._organism, "_loop", None)
            model = getattr(loop, "_world_self_dynamics", None) if loop else None
            style = OPERATION_STYLE.get(operation, "analytical")
            prediction = model.predict_from_context(style, context) if model and context else None
            if prediction is not None:
                cost = abs(float(prediction.self_deltas.get("cognitive_energy", cost)))
                gain = max(
                    0.0,
                    float(prediction.world_deltas.get("user_trust", 0.0))
                    + float(prediction.world_deltas.get("engagement_signal", 0.0))
                    + float(prediction.world_deltas.get("skill_depth_gain", 0.0)),
                )
                confidence = float(prediction.confidence)
                source = "world_self_dynamics"
        except Exception as exc:
            logger.debug("[IntentionReasoner] hypothesis prediction failed: %s", exc)

        score = ((gain + 0.02) * confidence) / max(cost, 0.01)
        return ReasoningHypothesis(
            operation=operation,
            statement=f"Using {operation} will produce grounded progress on the intention",
            expected_cost=round(cost, 4),
            expected_gain=round(gain, 4),
            confidence=round(confidence, 3),
            score=round(score, 4),
            evidence_source=source,
        )

    def _prune(self) -> None:
        if len(self._episodes) <= MAX_EPISODES:
            return
        oldest = sorted(self._episodes.values(), key=lambda item: item.updated_at)
        for episode in oldest[:len(self._episodes) - MAX_EPISODES]:
            self._episodes.pop(episode.id, None)

    def _save_locked(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": {key: asdict(value) for key, value in self._episodes.items()},
            "_meta": {"version": "v1", "updated_at": time.time()},
        }
        tmp = self._path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            for key, value in raw.get("episodes", {}).items():
                value["evidence"] = [ReasoningEvidence(**item) for item in value.get("evidence", [])]
                value["hypotheses"] = [
                    ReasoningHypothesis(**item) for item in value.get("hypotheses", [])
                ]
                value.setdefault("initial_confidence", value.get("confidence", 0.0))
                self._episodes[key] = ReasoningEpisode(**value)
        except Exception as exc:
            logger.warning("[IntentionReasoner] load failed: %s", exc)
