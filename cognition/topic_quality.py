"""
TopicQualityFilter
==================
Statistically-driven topic quality filter for the CuriosityEngine.

Instead of a static dictionary of stop words, this module scores topic
candidates using Inverse Document Frequency computed from the live
conversation corpus maintained by WorldModel.

Words that appear in many messages (high document frequency) have low IDF
and are function/filler words by definition — regardless of language.
Words that appear rarely (low DF) carry genuine information.

This approach is:
  - Language-agnostic (works for EN, FR, and any future language)
  - Self-updating (adapts as the conversation corpus grows)
  - Principled (mathematical basis rather than manual curation)

The filter also applies lightweight morphological heuristics to catch
verb conjugations and contractions that TF-IDF alone might miss in short
corpora (< 20 interactions).
"""

import math
import re
import unicodedata
from typing import List, Optional, Any


# ── Morphological patterns (language-agnostic heuristics) ────────────────────
# These catch verb forms and contractions before IDF scoring,
# useful when the corpus is too small for reliable IDF.

_VERB_SUFFIX_RE = re.compile(
    r"(?:"
    # French verb endings
    r"(ait|aient|ais|ez|ons|ant|ent|er|ait|ions|iez|ront|rait|raient|é|ée|és|ées)$"
    r"|"
    # English verb endings
    r"(ing|tion|tions|ment|ments|ness|ness|ical|ful|less|able|ible)$"
    r")",
    re.IGNORECASE | re.UNICODE,
)

# Characters that indicate this is a contraction, not a content word
_CONTRACTION_CHARS = set("''\u2019\u2018`")

# Minimum IDF score to accept a topic (below this = too common = function word)
# IDF = log((N+1)/(df+1)) + 1.0  — same formula as WorldModel
# At N=100 docs, df=30 → IDF = log(101/31)+1 = 2.19  (borderline)
# At N=100 docs, df=10 → IDF = log(101/11)+1 = 3.22  (good topic)
# At N=100 docs, df=50 → IDF = log(101/51)+1 = 1.68  (function word)
IDF_THRESHOLD = 2.0   # words with IDF < this are considered function words


class TopicQualityFilter:
    """
    Scores topic strings and decides whether they are worth adding to
    the CuriosityEngine's topic map.

    Usage:
        filter = TopicQualityFilter(world_model)
        if filter.is_valid(topic_string):
            curiosity.stimulate(topic_string, ...)
    """

    # Hard-coded bootstrap stop list — only the most universal function words
    # kept intentionally small; statistical filtering handles the rest
    _BOOTSTRAP_STOP: frozenset = frozenset({
        # English ultra-common
        "this", "that", "with", "from", "just", "have", "will", "would",
        "could", "should", "they", "them", "their", "been", "were", "what",
        "when", "where", "there", "here", "also", "then", "than", "very",
        "some", "more", "most", "such", "each", "both", "same", "only",
        "even", "still", "also", "thing", "things", "okay", "user", "system",
        "lumina", "response", "sentence", "speak", "about",
        # French ultra-common
        "dans", "avec", "pour", "mais", "donc", "ainsi", "alors", "aussi",
        "comme", "quand", "parce", "voila", "voici", "celle", "celui",
        "ceux", "cela", "ceci", "enfin", "apres", "avant",
        # French conversational fillers that survive IDF at low corpus sizes
        "bien", "tout", "fait", "doit", "peut", "etre", "avoir", "faire",
        "dire", "voir", "veux", "veut", "suis", "sont", "était", "sera",
        "merci", "votre", "notre", "autre", "entre", "selon", "depuis",
        "texte", "input", "chose", "rien", "quoi", "dont", "plus", "très",
        "assez", "trop", "peu", "beaucoup", "jamais", "toujours", "souvent",
        # ── Cognitive/meta noise words that dominate curiosity incorrectly ──
        # These are labels for cognitive *processes*, not content topics.
        # They appear constantly in internal thought formatting strings and
        # should never accumulate as curiosity topics.
        "statement", "question", "thought", "creating", "task_oriented",
        "task", "oriented", "recall", "middle", "currently", "dimly",
        "talking", "chase", "dive", "actually", "general", "unknown",
        "action", "goal", "memory", "insight", "belief", "reflection",
        "curiosity", "emotion", "tension", "context", "content", "source",
        "exploring", "understanding", "exploration", "continue", "verbe",
        "phrase", "topic", "concept", "word", "term", "type", "kind",
        "message", "prompt", "output", "result", "cycle", "tick", "loop",
        "internal", "external", "active", "passive", "current", "recent",
        "first", "second", "third", "last", "next", "previous", "another",
        "always", "never", "often", "maybe", "perhaps", "really", "quite",
        "already", "though", "however", "therefore", "because", "although",
        # Recommendation/filler forms are discourse acts, not durable topics.
        "recommend", "recommends", "recommended", "recommande", "recommander",
        "absolutely", "absolument",
        "actual", "currently", "actuel", "objectif",
    })

    def __init__(self, world_model: Optional[Any] = None):
        self._wm = world_model

    def attach(self, world_model: Any) -> None:
        """Call after world_model is available."""
        self._wm = world_model

    # ── Public API ────────────────────────────────────────────────────────────

    def is_valid(self, topic: str) -> bool:
        """Return True if topic is worthy of curiosity tracking."""
        return self.score(topic) >= 0.4

    def score(self, topic: str) -> float:
        """
        Return a quality score in [0, 1].
        0 = definitely a function word / noise
        1 = high-quality content topic
        """
        words = topic.lower().strip().split()
        if not words:
            return 0.0

        word_scores = [self._score_word(w) for w in words]
        return sum(word_scores) / len(word_scores)

    def filter_topics(self, topics: List[str]) -> List[str]:
        """Filter a list of candidate topics, returning only valid ones."""
        return [t for t in topics if self.is_valid(t)]

    # ── Internal scoring ──────────────────────────────────────────────────────

    def _score_word(self, word: str) -> float:
        """Score a single lowercase word token."""
        # 1. Too short
        if len(word) < 4:
            return 0.0

        # 2. Contains contraction characters
        if any(c in _CONTRACTION_CHARS for c in word):
            return 0.0

        # 3. Starts with French/English contraction prefix
        if re.match(r"^[jldstnmcJLDSTNMC]['\u2019\u2018]", word):
            return 0.0

        # 4. Bootstrap stop list (language-agnostic ultra-common words)
        normalized = unicodedata.normalize("NFD", word)
        ascii_approx = "".join(c for c in normalized if unicodedata.category(c) != "Mn")
        if ascii_approx in self._BOOTSTRAP_STOP or word in self._BOOTSTRAP_STOP:
            return 0.0

        # 5. Statistical IDF score (primary signal when corpus is large enough)
        idf_score = self._idf_score(word)
        if idf_score is not None:
            # Normalise: IDF_THRESHOLD → 0.0,  IDF=5.0 → 1.0
            idf_norm = max(0.0, min(1.0, (idf_score - IDF_THRESHOLD) / (5.0 - IDF_THRESHOLD)))
            if idf_norm < 0.05:
                return 0.0   # statistically proven to be a function word
            return idf_norm

        # 6. Morphological heuristics (corpus too small for reliable IDF)
        return self._morphological_score(word)

    def _idf_score(self, word: str) -> Optional[float]:
        """
        Return IDF score using WorldModel's live corpus, or None if corpus
        is too small (< 10 documents) to be statistically reliable.
        """
        if self._wm is None:
            return None
        try:
            total_docs = getattr(self._wm, '_total_docs', 0)
            if total_docs < 10:
                return None   # too small, fall back to morphological
            df = self._wm._global_topic_df.get(word, 0)
            idf = math.log((total_docs + 1) / (df + 1)) + 1.0
            return idf
        except Exception:
            return None

    def _morphological_score(self, word: str) -> float:
        """
        Heuristic scoring when IDF is unavailable.
        Returns a score in [0, 1] based on morphological properties.
        """
        # Penalise words that look like verb conjugations
        if _VERB_SUFFIX_RE.search(word) and len(word) < 8:
            return 0.15   # possible verb form — low but not zero

        # Prefer longer words (content words tend to be longer)
        length_score = min(1.0, (len(word) - 4) / 8.0)

        # Reward words with mixed consonant/vowel patterns (avoids acronyms)
        vowels = sum(1 for c in word.lower() if c in "aeiouyéèêëàâùûüîïô")
        vowel_ratio = vowels / len(word)
        balance_score = 1.0 - abs(vowel_ratio - 0.40) * 2   # ideal ~40% vowels

        return max(0.1, (length_score + max(0, balance_score)) / 2.0)


# ── Module-level singleton (lazy) ─────────────────────────────────────────────
_filter_instance: Optional[TopicQualityFilter] = None


def get_topic_filter(world_model: Optional[Any] = None) -> TopicQualityFilter:
    """Return the module singleton, optionally attaching a world_model."""
    global _filter_instance
    if _filter_instance is None:
        _filter_instance = TopicQualityFilter(world_model)
    elif world_model is not None:
        _filter_instance.attach(world_model)
    return _filter_instance
