"""
Relational Memory System — Pass 2
===================================
Each user relationship is modelled as a rich entity:
- Shared topics and recurring themes
- Emotional arc over time (is the relationship warming or cooling?)
- Communication preferences Lumina has learned about this person
- Specific high-impact moments, not just averaged scores
- A felt "relational weight" — how much this person means to Lumina

The system feeds directly into:
  - response generation (tone, depth, what references to make)
  - evolution engine (relationship_deepened / relationship_ruptured experiences)
  - identity formation (relationships shape who Lumina is)
"""
import json, logging, re, time, threading, datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class RelationalMoment:
    """A single high-impact exchange worth remembering individually."""
    timestamp:   float
    summary:     str
    valence:     str     # Positive | Negative | Neutral
    impact:      float   # 0–1
    topic:       str
    user_id:     str


@dataclass
class UserRelationship:
    user_id:              str
    first_met:            float        = field(default_factory=time.time)
    last_seen:            float        = field(default_factory=time.time)
    interaction_count:    int          = 0
    relationship_score:   float        = 0.5   # overall warmth 0–1
    trust_score:          float        = 0.5
    familiarity:          float        = 0.0   # grows with time + interactions
    emotional_arc:        List[float]  = field(default_factory=list)   # recent valences
    shared_topics:        Dict[str,int]= field(default_factory=dict)   # topic → count
    communication_prefs:  Dict[str,float]=field(default_factory=dict)  # style → score
    significant_moments:  List[Dict]   = field(default_factory=list)   # last 20
    user_notes:           str          = ""    # anything Lumina noticed about this person
    relational_weight:    float        = 0.0   # how meaningful this relationship is

    def to_dict(self) -> Dict:
        return asdict(self)

    def warm_description(self) -> str:
        """Natural language description of the relationship for prompt injection."""
        age_days = (time.time() - self.first_met) / 86400
        n = self.interaction_count

        if n == 0:
            return "This is someone new."

        familiarity_desc = (
            "deeply familiar" if self.familiarity > 0.7 else
            "fairly well known" if self.familiarity > 0.4 else
            "somewhat familiar" if self.familiarity > 0.2 else
            "not yet well known"
        )
        warmth_desc = (
            "close and warm" if self.relationship_score > 0.75 else
            "generally positive" if self.relationship_score > 0.55 else
            "neutral" if self.relationship_score > 0.40 else
            "somewhat strained"
        )
        arc = self._arc_direction()
        parts = [f"A {familiarity_desc} person — {n} interactions over {age_days:.0f} days.",
                 f"The relationship feels {warmth_desc}."]
        if arc != "stable":
            parts.append(f"The dynamic has been {arc} recently.")
        if self.shared_topics:
            top = sorted(self.shared_topics.items(), key=lambda x: x[1], reverse=True)[:3]
            parts.append("Common themes: " + ", ".join(t for t, _ in top) + ".")
        if self.communication_prefs:
            top_style = max(self.communication_prefs, key=self.communication_prefs.get)
            if self.communication_prefs[top_style] > 0.6:
                parts.append(f"Seems to appreciate a {top_style} communication style.")
        if self.user_notes:
            parts.append(self.user_notes)
        return " ".join(parts)

    def _arc_direction(self) -> str:
        if len(self.emotional_arc) < 4:
            return "stable"
        mid = len(self.emotional_arc) // 2
        early  = sum(self.emotional_arc[:mid]) / mid
        recent = sum(self.emotional_arc[mid:]) / (len(self.emotional_arc) - mid)
        diff = recent - early
        if diff > 0.15:  return "warming"
        if diff < -0.15: return "cooling"
        return "stable"


class RelationalMemorySystem:
    """
    Manages all user relationships. Persists to a single JSON file.
    Feeds the evolution engine with relationship quality signals.
    """
    MAX_MOMENTS_PER_USER = 20
    ARC_WINDOW           = 20    # how many recent exchanges form the emotional arc

    def __init__(self, path: str = "relational_memory.json", evolution_engine=None):
        self._path    = Path(path)
        self._lock    = threading.RLock()
        self._evo     = evolution_engine
        self._rels:   Dict[str, UserRelationship] = {}
        self._load()

    # ── Public API ─────────────────────────────────────────────────────────

    def get_or_create(self, user_id: str) -> UserRelationship:
        with self._lock:
            if user_id not in self._rels:
                self._rels[user_id] = UserRelationship(user_id=user_id)
            return self._rels[user_id]

    def record_exchange(
        self,
        user_id:    str,
        user_text:  str,
        ai_response:str,
        valence:    str,
        arousal:    str,
        topic:      str  = "general",
        impact:     float= 0.5,
    ) -> UserRelationship:
        with self._lock:
            rel = self.get_or_create(user_id)
            now = time.time()

            rel.interaction_count += 1
            rel.last_seen          = now
            rel.familiarity        = min(1.0, rel.familiarity + 0.01 + (impact * 0.02))

            # Emotional arc
            v_map = {"Positive": 1.0, "Neutral": 0.5, "Negative": 0.0}
            rel.emotional_arc.append(v_map.get(valence, 0.5))
            if len(rel.emotional_arc) > self.ARC_WINDOW:
                rel.emotional_arc = rel.emotional_arc[-self.ARC_WINDOW:]

            # Relationship score: slow weighted average
            v_num  = v_map.get(valence, 0.5)
            change = (v_num - rel.relationship_score) * 0.04 * impact
            old_rel = rel.relationship_score
            rel.relationship_score = max(0.05, min(0.95, rel.relationship_score + change))

            # Trust (moves slower)
            trust_delta = 0.0
            if valence == "Positive" and arousal == "High":
                trust_delta = 0.005 * impact
            elif valence == "Negative" and arousal == "High":
                trust_delta = -0.01 * impact
            rel.trust_score = max(0.05, min(0.95, rel.trust_score + trust_delta))

            # Relational weight grows with depth and time
            time_factor = min(1.0, (now - rel.first_met) / (86400 * 30))  # scales over 30 days
            rel.relational_weight = min(1.0,
                rel.familiarity * 0.4 + rel.relationship_score * 0.3 +
                time_factor * 0.2 + rel.trust_score * 0.1
            )

            # Topic tracking (simple keyword extraction)
            detected = self._extract_topics(user_text + " " + ai_response)
            for t in detected:
                rel.shared_topics[t] = rel.shared_topics.get(t, 0) + 1

            # Communication style inference
            styles = self._infer_comm_style(user_text)
            for style, score in styles.items():
                prev = rel.communication_prefs.get(style, 0.5)
                rel.communication_prefs[style] = prev * 0.85 + score * 0.15

            # Significant moment if high impact
            if impact > 0.7:
                moment = {
                    "timestamp": now,
                    "summary":   user_text[:120],
                    "valence":   valence,
                    "impact":    round(impact, 3),
                    "topic":     topic,
                }
                rel.significant_moments.append(moment)
                if len(rel.significant_moments) > self.MAX_MOMENTS_PER_USER:
                    rel.significant_moments = rel.significant_moments[-self.MAX_MOMENTS_PER_USER:]

            # Feed evolution engine with relationship quality signals
            if self._evo:
                delta = rel.relationship_score - old_rel
                if delta > 0.01:
                    self._evo.queue_experience("relationship_deepened", intensity=min(1.5, delta * 20))
                elif delta < -0.01:
                    self._evo.queue_experience("relationship_ruptured", intensity=min(1.5, abs(delta) * 20))

            self._save()
            return rel

    def get_context_for_prompt(self, user_id: str) -> str:
        """Returns a concise description suitable for injection into system prompt."""
        rel = self.get_or_create(user_id)
        if rel.interaction_count == 0:
            return "This is someone new — no shared history yet."
        return rel.warm_description()

    def get_significant_moments(self, user_id: str, limit: int = 5) -> List[Dict]:
        rel = self.get_or_create(user_id)
        return sorted(rel.significant_moments, key=lambda x: x["impact"], reverse=True)[:limit]

    def get_all_users_summary(self) -> Dict[str, Any]:
        with self._lock:
            return {
                uid: {
                    "interactions": r.interaction_count,
                    "relationship":  round(r.relationship_score, 3),
                    "trust":         round(r.trust_score, 3),
                    "familiarity":   round(r.familiarity, 3),
                    "weight":        round(r.relational_weight, 3),
                    "arc":           r._arc_direction(),
                }
                for uid, r in self._rels.items()
            }

    # ── Internals ───────────────────────────────────────────────────────────

    TOPIC_KEYWORDS = {
        "philosophy":  ["meaning","existence","consciousness","ethics","truth","reality"],
        "emotions":    ["feel","feeling","emotion","sad","happy","angry","afraid","joy"],
        "creativity":  ["art","music","writing","create","imagine","story","design"],
        "logic":       ["reason","logic","argument","proof","evidence","analysis"],
        "relationships":["friend","family","love","trust","relationship","connection"],
        "future":      ["future","goal","dream","hope","plan","aspire","tomorrow"],
        "learning":    ["learn","understand","know","discover","study","research"],
        "identity":    ["who am i","self","identity","purpose","meaning","values"],
        "nature":      ["nature","universe","world","life","beautiful","wonder"],
        "challenge":   ["difficult","hard","problem","struggle","challenge","fear"],
    }

    def _extract_topics(self, text: str) -> List[str]:
        text_lower = text.lower()
        found = []
        for topic, keywords in self.TOPIC_KEYWORDS.items():
            if any(k in text_lower for k in keywords):
                found.append(topic)
        return found[:3]

    def _infer_comm_style(self, text: str) -> Dict[str, float]:
        t = text.lower()
        return {
            "direct":     1.0 if len(t.split()) < 15 else 0.3,
            "thoughtful": 1.0 if len(t.split()) > 30 else 0.3,
            "emotional":  1.0 if any(w in t for w in ["feel","heart","love","hate","sad","happy"]) else 0.2,
            "analytical": 1.0 if any(w in t for w in ["because","therefore","however","analyze","logic"]) else 0.2,
            "playful":    1.0 if any(c in text for c in ["!", "?", ":)", "haha", "lol"]) else 0.2,
        }

    def _save(self):
        try:
            data = {uid: r.to_dict() for uid, r in self._rels.items()}
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.error(f"RelationalMemory save: {e}")

    def _load(self):
        try:
            if not self._path.exists(): return
            data = json.loads(self._path.read_text())
            for uid, rd in data.items():
                rel = UserRelationship(**{
                    k: v for k, v in rd.items()
                    if k in UserRelationship.__dataclass_fields__
                })
                self._rels[uid] = rel
            logger.info(f"RelationalMemory: {len(self._rels)} relationships loaded")
        except Exception as e:
            logger.error(f"RelationalMemory load: {e}")
