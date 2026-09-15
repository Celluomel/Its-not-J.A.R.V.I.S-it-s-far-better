"""
WorldModel — Extended World Representation
===========================================
Extensions over the minimal version:
  1. Topic extraction : TF-IDF over real interaction history + bigrams
  2. Expertise scoring: evolving score with decay and confirmation
  3. Auto-learned causal beliefs from temporal co-occurrence
  4. Temporal patterns: active hours, rhythm, silence gaps
  5. Relational dynamics: engagement trajectory
  6. predict_from_context: lexical similarity scoring
  7. Enriched prompt_fragment

Persists to: data/world_model.json
"""

import json
import logging
import math
import re
import time
from managers.settings_manager import get_persona_name as _gpn

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# FR + EN stop words
_STOP_WORDS = {
    "the","a","an","is","are","was","were","be","been","being","have","has","had",
    "do","does","did","will","would","could","should","may","might","can","need",
    "i","you","he","she","it","we","they","me","him","her","us","them","my","your",
    "his","its","our","their","this","that","these","those","and","but","or","nor",
    "for","yet","so","if","then","when","where","why","how","what","who","which",
    "in","on","at","by","with","about","into","through","from","up","down","of",
    "to","not","no","yes","just","also","than","as","very","here","there","now",
    "get","got","make","like","know","think","want","come","look","see","feel",
    # Common conversational words that leak through as false topics
    "concept","without","inner","some","more","other","any","all","each","both",
    "few","many","much","most","own","such","same","because","while","after",
    "before","between","during","against","since","until","would","could","really",
    "something","anything","everything","nothing","someone","thing","things",
    "way","ways","lot","kind","type","part","point","time","times","place","use",
    "used","using","say","said","says","tell","told","ask","asked","asking","mean",
    "means","give","take","let","put","try","tried","though","even","rather","quite",
    "well","already","always","never","still","often","sometimes","maybe","perhaps",
    "sure","actually","usually","especially","instead","however","therefore",
    "response","lumina","user","system","message","chat","said","okay","okay",
    # French
    "le","la","les","un","une","des","du","de","et","en","a","au","aux","je","tu",
    "il","elle","nous","vous","ils","elles","me","te","se","lui","mon","ma","mes",
    "ton","ta","tes","son","sa","ses","notre","votre","leur","ce","cet","cette",
    "ces","qui","que","quoi","dont","ou","si","ne","pas","plus","tres","bien",
    "aussi","meme","encore","toujours","jamais","etre","avoir","faire","dire",
    "voir","vouloir","pouvoir","aller","venir","est","sont","etait","ont","avec",
    "dans","sur","sous","par","pour","mais","donc","car","ni","quand","comme",
    "puis","alors","ainsi","tout","tous","toute","toutes","autre","autres",
    "cela","ceci","ca","ici","la","ya","peu","beaucoup","trop","assez","tres",
}


@dataclass
class TopicEntry:
    name:        str
    raw_count:   int   = 0
    tfidf_score: float = 0.0
    last_seen:   float = field(default_factory=time.time)
    first_seen:  float = field(default_factory=time.time)


@dataclass
class ExpertiseSignal:
    domain:        str
    score:         float = 0.30
    confirmations: int   = 0
    last_signal:   float = field(default_factory=time.time)
    evidence:      List[str] = field(default_factory=list)

    def strengthen(self, amount: float = 0.08, phrase: str = "") -> None:
        self.score = min(0.95, self.score + amount * (1.0 - self.score))
        self.confirmations += 1
        self.last_signal = time.time()
        if phrase:
            self.evidence.append(phrase[:60])
            self.evidence = self.evidence[-5:]

    def decay(self, rate: float = 0.015) -> None:
        age_days = (time.time() - self.last_signal) / 86400
        if age_days > 7:
            self.score = max(0.10, self.score - rate * age_days)

    def label(self) -> str:
        if self.score > 0.75: return "expert"
        if self.score > 0.55: return "proficient"
        if self.score > 0.35: return "familiar"
        return "novice"


@dataclass
class TemporalPattern:
    active_hours:    Dict[int, int]   = field(default_factory=dict)
    active_days:     Dict[int, int]   = field(default_factory=dict)
    avg_gap_seconds: float            = 0.0
    last_active:     float            = field(default_factory=time.time)
    _gap_samples:    List[float]      = field(default_factory=list)

    def record(self, now: float) -> None:
        dt = datetime.fromtimestamp(now)
        self.active_hours[dt.hour]     = self.active_hours.get(dt.hour, 0) + 1
        self.active_days[dt.weekday()] = self.active_days.get(dt.weekday(), 0) + 1
        if self.last_active > 0:
            gap = now - self.last_active
            if gap < 3600:
                self._gap_samples.append(gap)
                self._gap_samples = self._gap_samples[-20:]
                if self._gap_samples:
                    self.avg_gap_seconds = sum(self._gap_samples) / len(self._gap_samples)
        self.last_active = now

    def peak_hour(self) -> Optional[int]:
        if not self.active_hours:
            return None
        return max(self.active_hours, key=self.active_hours.get)


@dataclass
class RelationalDynamic:
    engagement_scores:   List[float] = field(default_factory=list)
    avg_message_length:  float = 0.0
    question_ratio:      float = 0.0
    formality_score:     float = 0.5
    trust_signals:       int   = 0
    frustration_signals: int   = 0

    def record_interaction(self, user_text: str, response_len: int) -> None:
        n = len(user_text.split())
        engagement = min(1.0, n / 30.0 + (0.2 if "?" in user_text else 0))
        self.engagement_scores.append(engagement)
        self.engagement_scores = self.engagement_scores[-20:]
        self.avg_message_length = self.avg_message_length * 0.85 + n * 0.15
        self.question_ratio     = self.question_ratio * 0.9 + (0.1 if "?" in user_text else 0)

        text_l = user_text.lower()
        formal  = sum(1 for m in ("please","thank","could you","merci","veuillez") if m in text_l)
        casual  = sum(1 for m in ("hey","yeah","ok","ouais","salut","cool") if m in text_l)
        if formal > casual:
            self.formality_score = min(1.0, self.formality_score + 0.05)
        elif casual > formal:
            self.formality_score = max(0.0, self.formality_score - 0.05)

        trust = sum(1 for w in ("thanks","merci","great","parfait","excellent") if w in text_l)
        frust = sum(1 for w in ("wrong","incorrect","broken","ne fonctionne","erreur") if w in text_l)
        self.trust_signals       += trust
        self.frustration_signals += frust

    def engagement_trend(self) -> str:
        if len(self.engagement_scores) < 4:
            return "unknown"
        mid = len(self.engagement_scores) // 2
        first  = sum(self.engagement_scores[:mid]) / mid
        second = sum(self.engagement_scores[mid:]) / (len(self.engagement_scores) - mid)
        if second - first > 0.08:  return "rising"
        if first - second > 0.08:  return "declining"
        return "stable"

    def relationship_label(self) -> str:
        recent = self.engagement_scores[-5:] if self.engagement_scores else []
        avg = sum(recent) / max(1, len(recent))
        if self.trust_signals > 5 and avg > 0.5: return "close"
        if avg > 0.4:  return "engaged"
        if avg < 0.15: return "distant"
        return "neutral"


@dataclass
class CausalBelief:
    antecedent:   str
    consequent:   str
    confidence:   float = 0.5
    count:        int   = 1
    last_seen:    float = field(default_factory=time.time)
    auto_learned: bool  = False

    def strengthen(self, amount: float = 0.06) -> None:
        self.confidence = min(0.95, self.confidence + amount * (1.0 - self.confidence))
        self.count     += 1
        self.last_seen  = time.time()

    def weaken(self, amount: float = 0.04) -> None:
        self.confidence = max(0.05, self.confidence - amount)


@dataclass
class UserProfile:
    user_id:            str
    display_name:       str   = ""
    first_seen:         float = field(default_factory=time.time)
    last_seen:          float = field(default_factory=time.time)
    interaction_count:  int   = 0
    topic_entries:      Dict[str, TopicEntry]    = field(default_factory=dict)
    expertise:          Dict[str, ExpertiseSignal] = field(default_factory=dict)
    preferred_depth:    str   = "medium"
    formality:          str   = "neutral"
    avg_message_words:  float = 0.0
    temporal:           Optional[TemporalPattern]   = field(default=None)
    relational:         Optional[RelationalDynamic] = field(default=None)
    _recent_topic_windows: List[List[str]]          = field(default_factory=list)
    _topic_quality:        float                     = 1.0   # 0–1, semantic vs surface
    # Emergent vocabulary map — learned per-user, per-tier
    # {tier: {phrase: weight}}  e.g. {"visual": {"la pièce": 1.4, "what you see": 0.9}}
    vocab_map: Dict[str, Dict[str, float]]           = field(default_factory=lambda: {
        "visual":      {},
        "event":       {},
        "cognitive":   {},
        "interaction": {},
    })

    def __post_init__(self):
        if self.temporal  is None: self.temporal  = TemporalPattern()
        if self.relational is None: self.relational = RelationalDynamic()

    def top_topics(self, n: int = 5) -> List[str]:
        now = time.time()
        scored = []
        for name, entry in self.topic_entries.items():
            recency = math.exp(-(now - entry.last_seen) / (14 * 86400))
            scored.append((name, entry.tfidf_score * recency + entry.raw_count * 0.05))
        return [t for t, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:n]]

    def top_expertise(self, n: int = 3) -> List[Tuple[str, str]]:
        return [
            (d, e.label())
            for d, e in sorted(self.expertise.items(), key=lambda kv: kv[1].score, reverse=True)[:n]
            if e.score > 0.35
        ]

    def prompt_fragment(self) -> str:
        parts = []
        name = self.display_name or self.user_id
        parts.append(f"User model based on {self.interaction_count} observed exchanges (patterns are tentative).")
        if name and name not in ("default", "guest"):
            parts.append(f"User: {name}")

        topics = self.top_topics(3)
        if topics:
            parts.append("Topics discussed: " + ", ".join(topics))

        exp = self.top_expertise(2)
        if exp:
            parts.append("Possible familiarity signals (inferred from vocabulary, unverified): " +
                         ", ".join(f"{d} ({l})" for d, l in exp))

        style = []
        if self.preferred_depth != "medium": style.append(self.preferred_depth)
        if self.formality != "neutral":      style.append(self.formality)
        if style: parts.append("Style: " + ", ".join(style))

        if self.relational and self.interaction_count >= 3:
            lbl   = self.relational.relationship_label()
            trend = self.relational.engagement_trend()
            if lbl in ("close", "distant"):     parts.append(f"Rel: {lbl}")
            if trend in ("rising", "declining"): parts.append(f"Trend: {trend}")

        if self.temporal:
            ph = self.temporal.peak_hour()
            if ph is not None:
                session = "morning" if 5<=ph<12 else "afternoon" if 12<=ph<18 else "evening" if 18<=ph<23 else "night"
                parts.append(f"Active: {session}")

        parts.append(f"{self.interaction_count} interactions")
        return " | ".join(parts)


class WorldModel:

    MAX_USERS          = 50
    MAX_CAUSAL_BELIEFS = 200
    TOPIC_MAX_PER_USER = 80
    TFIDF_UPDATE_EVERY = 5
    CAUSAL_MIN_COUNT   = 3

    _DOMAIN_VOCAB: Dict[str, List[str]] = {
        "programming":    ["function","class","method","variable","loop","array","algorithm",
                           "debug","compile","runtime","syntax","library","framework","api",
                           "async","thread","memory","recursion","inheritance","python",
                           "javascript","typescript","rust","golang","kotlin","def ","import "],
        "machine_learning":["training","inference","dataset","overfitting","gradient",
                            "backprop","neural","embedding","transformer","attention","loss",
                            "batch","epoch","weight","activation","layer","convolution","lstm",
                            "fine-tuning","rag","tokenizer","supervised","unsupervised"],
        "mathematics":    ["theorem","proof","integral","derivative","matrix","vector","tensor",
                           "eigenvalue","topology","manifold","probability","distribution",
                           "variance","stochastic","linear algebra","differential","polynomial"],
        "philosophy":     ["consciousness","qualia","ontology","epistemology","phenomenology",
                           "determinism","free will","ethics","moral","existential","metaphysics",
                           "empiricism","rationalism","dialectic","intentionality","emergence"],
        "science":        ["hypothesis","experiment","empirical","falsifiable","quantum",
                           "relativity","entropy","thermodynamics","evolution","dna","protein",
                           "neuron","photon","electron","nucleus"],
        "systems_design": ["architecture","microservice","monolith","scaling","latency",
                           "throughput","cache","sharding","replication","load balancer",
                           "container","kubernetes","docker","ci/cd","serverless","cloud"],
    }

    def __init__(self, organism: Any, path: str = "data/world_model.json"):
        self._o    = organism
        self._path = Path(path)
        self._lock = Lock()

        self.user_profiles:  Dict[str, UserProfile] = {}
        self.topic_graph:    Dict[str, Counter]     = defaultdict(Counter)
        self.causal_beliefs: List[CausalBelief]     = []
        self._global_topic_df: Counter              = Counter()
        self._total_docs: int                       = 0

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._load()
        logger.info(
            f"[WorldModel] loaded: {len(self.user_profiles)} users, "
            f"{len(self.causal_beliefs)} causal beliefs"
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def update_from_interaction(
        self, user_id: str, user_text: str, response: str, emotion: str = "neutral"
    ) -> UserProfile:
        now = time.time()
        with self._lock:
            profile = self.user_profiles.setdefault(user_id, UserProfile(user_id=user_id))
            profile.interaction_count += 1
            profile.last_seen          = now

            user_topics = self._extract_topics(user_text)
            resp_topics = self._extract_topics(response)
            all_topics  = list(set(user_topics + resp_topics))

            self._update_topic_entries(profile, user_topics, resp_topics)

            self._total_docs += 1
            for t in set(all_topics):
                self._global_topic_df[t] += 1

            if profile.interaction_count % self.TFIDF_UPDATE_EVERY == 0:
                self._recalculate_tfidf(profile)

            self._update_expertise(profile, user_text)
            self._update_style(profile, user_text)

            profile.temporal.record(now)
            profile.relational.record_interaction(user_text, len(response.split()))

            if len(all_topics) >= 2:
                for i, t1 in enumerate(all_topics):
                    for t2 in all_topics[i+1:]:
                        self.topic_graph[t1][t2] += 1
                        self.topic_graph[t2][t1] += 1

            profile._recent_topic_windows.append(user_topics)
            profile._recent_topic_windows = profile._recent_topic_windows[-10:]
            self._auto_learn_causal(profile)

            if len(self.user_profiles) > self.MAX_USERS:
                oldest = min(self.user_profiles.items(), key=lambda kv: kv[1].last_seen)
                del self.user_profiles[oldest[0]]

            if profile.interaction_count % 10 == 0:
                for exp in profile.expertise.values():
                    exp.decay()

            self._save()
            return profile

    def learn_vocab(
        self,
        user_id:   str,
        phrases:   List[str],
        tier:      str,
        strength:  float = 0.3,
    ):
        """
        Reinforce that these phrases (from user's message) are associated
        with memory tier 'tier'. Called when a retrieval for that tier
        succeeds and produces a used memory.
        strength: how much to boost (0.1 weak signal, 0.5 strong correction)
        """
        with self._lock:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return
            tier_map = profile.vocab_map.setdefault(tier, {})
            for phrase in phrases:
                phrase = phrase.lower().strip()
                if len(phrase) < 3:
                    continue
                current = tier_map.get(phrase, 0.0)
                # EMA update — existing weight anchors new signal
                tier_map[phrase] = min(2.0, current * 0.8 + strength)
            # Decay all weights slightly — unused phrases fade
            for p in list(tier_map.keys()):
                tier_map[p] *= 0.995
                if tier_map[p] < 0.05:
                    del tier_map[p]
            self._save()

    def get_vocab_tier(self, user_id: str, text: str) -> Optional[str]:
        """
        Given a user's message, return the most likely memory tier
        based on their learned vocabulary, or None if no signal.
        Threshold: combined weight > 0.4 to avoid false positives.
        """
        profile = self.user_profiles.get(user_id)
        if not profile:
            return None
        text_lower = text.lower()
        scores: Dict[str, float] = {}
        for tier, vocab in profile.vocab_map.items():
            total = sum(
                weight for phrase, weight in vocab.items()
                if phrase in text_lower
            )
            if total > 0:
                scores[tier] = total
        if not scores:
            return None
        best_tier  = max(scores, key=lambda k: scores[k])
        best_score = scores[best_tier]
        return best_tier if best_score >= 0.4 else None

    def get_user(self, user_id: str) -> Optional[UserProfile]:
        return self.user_profiles.get(user_id)

    def user_prompt_fragment(self, user_id: str) -> str:
        profile = self.get_user(user_id)
        if not profile or profile.interaction_count < 2:
            return ""
        return f"[User model] {profile.prompt_fragment()}"

    def predict_from_context(self, context: str) -> List[Tuple[str, float]]:
        context_tokens = set(self._tokenize(context))
        if not context_tokens:
            return []
        matches = []
        for belief in self.causal_beliefs:
            ant_tokens = set(self._tokenize(belief.antecedent))
            if not ant_tokens:
                continue
            overlap = len(context_tokens & ant_tokens) / len(ant_tokens)
            if overlap >= 0.4:
                matches.append((belief.consequent, round(overlap * belief.confidence, 3)))
        return sorted(matches, key=lambda x: x[1], reverse=True)[:4]

    def related_topics(self, topic: str, n: int = 4) -> List[str]:
        if topic not in self.topic_graph:
            return []
        return [t for t, _ in self.topic_graph[topic].most_common(n)]

    def learn_causal(self, antecedent: str, consequent: str, confidence: float = 0.5) -> None:
        with self._lock:
            for b in self.causal_beliefs:
                if b.antecedent.lower() == antecedent.lower() and b.consequent.lower() == consequent.lower():
                    b.strengthen()
                    return
            if len(self.causal_beliefs) < self.MAX_CAUSAL_BELIEFS:
                self.causal_beliefs.append(CausalBelief(antecedent=antecedent, consequent=consequent, confidence=confidence))
                self._save()

    def has_contradiction(self, goal_or_topic: str, confidence_threshold: float = 0.4) -> bool:
        """
        Gemini's detect_friction integration: check whether any causal belief
        contradicts the given goal/topic. A contradiction exists when a belief
        asserts a consequent that is semantically opposite to the goal.

        Simple token-overlap heuristic — returns True if a belief's antecedent
        overlaps with the goal AND its confidence is below threshold (uncertain
        or contested belief that conflicts with proposed action).

        Used by contradiction_handler.detect_contradiction() to score
        world-model-level friction, not just text-pattern friction.
        """
        goal_tokens = set(goal_or_topic.lower().split())
        negation_words = {"not", "never", "no", "cannot", "cant", "wont", "avoid",
                          "dislike", "reject", "refuse", "impossible", "unable"}

        with self._lock:
            for belief in self.causal_beliefs:
                ant_tokens = set(belief.antecedent.lower().split())
                overlap = goal_tokens & ant_tokens

                if not overlap:
                    continue

                # If belief contains negation AND overlaps with goal → contradiction
                if negation_words & ant_tokens and belief.confidence >= confidence_threshold:
                    return True

                # Low-confidence belief about this topic → epistemic friction
                if belief.confidence < 0.25 and len(overlap) >= 2:
                    return True

        return False

    def summary(self) -> Dict:
        return {
            "users":           len(self.user_profiles),
            "causal_beliefs":  len(self.causal_beliefs),
            "auto_causal":     sum(1 for b in self.causal_beliefs if b.auto_learned),
            "topic_nodes":     len(self.topic_graph),
            "total_docs_seen": self._total_docs,
        }

    # ── Topic Extraction ──────────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        tokens = re.findall(r"[a-zA-ZÀ-ÿ’']{3,}", text.lower())
        return [t for t in tokens if t not in _STOP_WORDS]

    def _extract_topics(self, text: str) -> List[str]:
        topics = []
        tokens = self._tokenize(text)
        for t in tokens:
            if len(t) >= 4:
                topics.append(t)
        for i in range(len(tokens) - 1):
            bigram = tokens[i] + "_" + tokens[i+1]
            if len(bigram) > 8:
                topics.append(bigram)
        # Capitalised proper nouns
        for w in text.split():
            clean = w.strip('.,!?"()[]:-')
            if len(clean) > 3 and clean[0].isupper() and clean.lower() not in _STOP_WORDS and clean not in (_gpn(),"I","The","A","An","Je","La","Le"):
                topics.append(clean.lower())
        seen, result = set(), []
        for t in topics:
            if t not in seen:
                seen.add(t)
                result.append(t)
        return result[:12]

    def _update_topic_entries(self, profile: UserProfile, user_topics: List[str], resp_topics: List[str]) -> None:
        now = time.time()
        for t in user_topics:
            if t not in profile.topic_entries:
                profile.topic_entries[t] = TopicEntry(name=t, first_seen=now)
            profile.topic_entries[t].raw_count += 2
            profile.topic_entries[t].last_seen  = now
        for t in resp_topics:
            if t not in profile.topic_entries:
                profile.topic_entries[t] = TopicEntry(name=t, first_seen=now)
            profile.topic_entries[t].raw_count += 1
        if len(profile.topic_entries) > self.TOPIC_MAX_PER_USER:
            sorted_e = sorted(profile.topic_entries.items(), key=lambda kv: (kv[1].last_seen, kv[1].raw_count))
            for name, _ in sorted_e[:10]:
                del profile.topic_entries[name]

    # Topics with TF-IDF below this are pruned as noise
    TFIDF_PRUNE_THRESHOLD = 0.002
    # Max topics per user profile
    MAX_PROFILE_TOPICS    = 300

    def _recalculate_tfidf(self, profile: UserProfile) -> None:
        # ── Active forgetting: decay counts so stale topics fade ──────────
        now = time.time()
        decay_per_hour = 0.997  # lose 0.3% per hour → forgotten in ~14 days
        for entry in profile.topic_entries.values():
            age_hours = (now - entry.last_seen) / 3600.0
            if age_hours > 1:
                entry.raw_count = max(1, entry.raw_count * (decay_per_hour ** age_hours))

        # ── TF-IDF scoring ─────────────────────────────────────────────────
        total = max(1, sum(e.raw_count for e in profile.topic_entries.values()))
        for name, entry in profile.topic_entries.items():
            tf  = entry.raw_count / total
            df  = self._global_topic_df.get(name, 1)
            idf = math.log((self._total_docs + 1) / (df + 1)) + 1.0
            entry.tfidf_score = round(tf * idf, 4)

        # ── Prune noise topics ─────────────────────────────────────────────
        noise = [n for n, e in profile.topic_entries.items()
                 if e.tfidf_score < self.TFIDF_PRUNE_THRESHOLD and e.raw_count < 3]
        for n in noise:
            del profile.topic_entries[n]

        # ── Trim to MAX_PROFILE_TOPICS if needed ───────────────────────────
        if len(profile.topic_entries) > self.MAX_PROFILE_TOPICS:
            sorted_topics = sorted(
                profile.topic_entries.items(),
                key=lambda kv: kv[1].tfidf_score, reverse=True
            )
            keep = {k for k, _ in sorted_topics[:self.MAX_PROFILE_TOPICS]}
            for k in list(profile.topic_entries.keys()):
                if k not in keep:
                    del profile.topic_entries[k]

        # ── Topic quality metric (semantic vs surface tokens) ──────────────
        total_t  = len(profile.topic_entries)
        # Surface tokens: short, no underscore, no uppercase origin
        surface  = sum(1 for n in profile.topic_entries
                       if len(n) <= 5 and '_' not in n)
        profile._topic_quality = round(1.0 - surface / max(1, total_t), 3)

    def _update_expertise(self, profile: UserProfile, user_text: str) -> None:
        text_lower = user_text.lower()
        for domain, vocab in self._DOMAIN_VOCAB.items():
            hits    = sum(1 for v in vocab if v in text_lower)
            density = hits / max(1, len(user_text.split()))
            if hits == 0:
                continue
            exp = profile.expertise.setdefault(domain, ExpertiseSignal(domain=domain))
            amount = 0.12 if hits >= 4 else 0.07 if hits >= 2 else 0.03
            if density > 0.10:
                amount = min(0.15, amount * 1.5)
            phrase = ""
            for v in vocab:
                if v in text_lower:
                    idx = text_lower.find(v)
                    phrase = user_text[max(0, idx-10):idx+len(v)+10].strip()
                    break
            exp.strengthen(amount=amount, phrase=phrase)

    def _update_style(self, profile: UserProfile, user_text: str) -> None:
        n = len(user_text.split())
        profile.avg_message_words = profile.avg_message_words * 0.85 + n * 0.15
        avg = profile.avg_message_words
        profile.preferred_depth = "brief" if avg < 8 else "deep" if avg > 35 else "medium"
        if profile.relational:
            fs = profile.relational.formality_score
            profile.formality = "formal" if fs > 0.6 else "casual" if fs < 0.35 else "neutral"

    def _auto_learn_causal(self, profile: UserProfile) -> None:
        windows = profile._recent_topic_windows
        if len(windows) < self.CAUSAL_MIN_COUNT + 1:
            return
        transitions: Counter = Counter()
        for i in range(len(windows) - 1):
            for a in windows[i]:
                for b in windows[i+1]:
                    if a != b and len(a) > 3 and len(b) > 3:
                        transitions[(a, b)] += 1
        for (a, b), count in transitions.items():
            if count >= self.CAUSAL_MIN_COUNT:
                exists = any(bl.antecedent == a and bl.consequent == b for bl in self.causal_beliefs)
                if exists:
                    for bl in self.causal_beliefs:
                        if bl.antecedent == a and bl.consequent == b:
                            bl.strengthen(0.03)
                elif len(self.causal_beliefs) < self.MAX_CAUSAL_BELIEFS:
                    self.causal_beliefs.append(CausalBelief(
                        antecedent=a, consequent=b,
                        confidence=min(0.70, 0.35 + count * 0.05),
                        count=count, auto_learned=True,
                    ))
                    logger.debug(f"[WorldModel] auto-causal: {a!r} -> {b!r} (count={count})")

    # ── Persistence ───────────────────────────────────────────────────────────

    def _save(self) -> None:
        try:
            profiles_data = {}
            for uid, p in self.user_profiles.items():
                profiles_data[uid] = {
                    "user_id": p.user_id, "display_name": p.display_name,
                    "first_seen": p.first_seen, "last_seen": p.last_seen,
                    "interaction_count": p.interaction_count,
                    "preferred_depth": p.preferred_depth, "formality": p.formality,
                    "avg_message_words": p.avg_message_words,
                    "topic_entries": {
                        n: {"name": e.name, "raw_count": e.raw_count, "tfidf_score": e.tfidf_score,
                            "last_seen": e.last_seen, "first_seen": e.first_seen}
                        for n, e in list(p.topic_entries.items())[:60]
                    },
                    "expertise": {
                        d: {"domain": ex.domain, "score": ex.score, "confirmations": ex.confirmations,
                            "last_signal": ex.last_signal, "evidence": ex.evidence}
                        for d, ex in p.expertise.items()
                    },
                    "temporal": {
                        "active_hours": p.temporal.active_hours if p.temporal else {},
                        "active_days": p.temporal.active_days if p.temporal else {},
                        "avg_gap_seconds": p.temporal.avg_gap_seconds if p.temporal else 0.0,
                        "last_active": p.temporal.last_active if p.temporal else 0.0,
                        "_gap_samples": p.temporal._gap_samples if p.temporal else [],
                    },
                    "relational": {
                        "engagement_scores": (p.relational.engagement_scores[-20:] if p.relational else []),
                        "avg_message_length": (p.relational.avg_message_length if p.relational else 0.0),
                        "question_ratio": (p.relational.question_ratio if p.relational else 0.0),
                        "formality_score": (p.relational.formality_score if p.relational else 0.5),
                        "trust_signals": (p.relational.trust_signals if p.relational else 0),
                        "frustration_signals": (p.relational.frustration_signals if p.relational else 0),
                    },
                    "_recent_topic_windows": p._recent_topic_windows[-10:],
                    "vocab_map": p.vocab_map,
                }
            data = {
                "user_profiles": profiles_data,
                "causal_beliefs": [asdict(b) for b in self.causal_beliefs[-100:]],
                "topic_graph": {k: dict(v.most_common(20)) for k, v in list(self.topic_graph.items())[:100]},
                "global_topic_df": dict(self._global_topic_df.most_common(200)),
                "total_docs": self._total_docs,
            }
            self._path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug(f"[WorldModel] save error: {e}")

    def _load(self) -> None:
        # Migration (re-applied — regressed after mid-session recovery from
        # an older zip snapshot): the path bug fixed in cognitive_organism.py
        # meant this file was previously written one directory up (e.g.
        # data/world_model.json instead of data/persona/world_model.json).
        # On first load after the fix, if the new path doesn't exist yet but
        # the old mis-pathed file does, copy it forward once so accumulated
        # causal beliefs and user profiles aren't silently lost. Harmless
        # no-op once migration has happened (new path exists from then on).
        if not self._path.exists():
            try:
                _old_path = self._path.parent.parent / self._path.name
                if _old_path.exists() and _old_path != self._path:
                    import shutil
                    self._path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(_old_path, self._path)
                    logger.info(
                        f"[WorldModel] Migrated existing data from old path "
                        f"{_old_path} → {self._path}"
                    )
            except Exception as _mig_e:
                logger.debug(f"[WorldModel] migration check failed: {_mig_e}")

        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            for uid, pd in data.get("user_profiles", {}).items():
                p = UserProfile(
                    user_id=pd["user_id"], display_name=pd.get("display_name",""),
                    first_seen=pd.get("first_seen", time.time()), last_seen=pd.get("last_seen", time.time()),
                    interaction_count=pd.get("interaction_count", 0),
                    vocab_map=pd.get("vocab_map", {"visual":{},"event":{},"cognitive":{},"interaction":{}}),
                    preferred_depth=pd.get("preferred_depth","medium"),
                    formality=pd.get("formality","neutral"),
                    avg_message_words=pd.get("avg_message_words", 0.0),
                )
                for name, ed in pd.get("topic_entries", {}).items():
                    p.topic_entries[name] = TopicEntry(**ed)
                for domain, exd in pd.get("expertise", {}).items():
                    p.expertise[domain] = ExpertiseSignal(**exd)
                td = pd.get("temporal", {})
                p.temporal = TemporalPattern(
                    active_hours={int(k): v for k, v in td.get("active_hours", {}).items()},
                    active_days={int(k): v for k, v in td.get("active_days", {}).items()},
                    avg_gap_seconds=td.get("avg_gap_seconds", 0.0),
                    last_active=td.get("last_active", time.time()),
                    _gap_samples=td.get("_gap_samples", []),
                )
                rd = pd.get("relational", {})
                p.relational = RelationalDynamic(
                    engagement_scores=rd.get("engagement_scores", []),
                    avg_message_length=rd.get("avg_message_length", 0.0),
                    question_ratio=rd.get("question_ratio", 0.0),
                    formality_score=rd.get("formality_score", 0.5),
                    trust_signals=rd.get("trust_signals", 0),
                    frustration_signals=rd.get("frustration_signals", 0),
                )
                p._recent_topic_windows = pd.get("_recent_topic_windows", [])
                self.user_profiles[uid] = p
            for bd in data.get("causal_beliefs", []):
                self.causal_beliefs.append(CausalBelief(**bd))
            for topic, links in data.get("topic_graph", {}).items():
                self.topic_graph[topic] = Counter(links)
            self._global_topic_df = Counter(data.get("global_topic_df", {}))
            self._total_docs      = data.get("total_docs", 0)
        except Exception as e:
            logger.warning(f"[WorldModel] load error: {e}")
