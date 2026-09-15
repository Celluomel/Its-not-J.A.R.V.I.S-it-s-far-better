"""
cognition/narrative_compression.py  (v63)

Compresses accumulated life chapters into coherent identity themes.
Prevents identity from becoming an unbounded growing list.

Every COMPRESS_EVERY_N slow cycles:
1. Read all life chapters from NarrativeIdentity
2. Cluster by semantic similarity (keyword overlap + topic co-occurrence)
3. For each cluster, synthesise a theme: a single sentence that captures
   what this cluster of experiences means for identity
4. Update NarrativeIdentity._current_arc to reflect top themes
5. Prune chapters with significance < PRUNE_THRESHOLD that are covered
   by a theme (the theme captures their content — the chapter is redundant)

This is not memory deletion.  The chapter data remains in FAISS.
What is pruned is the in-memory life_story list that gets read into
prompts — the difference between "a 200-item log" and "five coherent themes
with supporting chapters."
"""

from __future__ import annotations
import json, logging, re, threading, time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

COMPRESS_EVERY_N  = 80
PRUNE_THRESHOLD   = 0.35    # significance below this = prunable if covered by theme
MIN_CHAPTERS      = 10      # don't compress until enough data
SAVE_PATH         = "data/persona/narrative_compression.json"


@dataclass
class IdentityTheme:
    theme:       str    # synthesised theme sentence
    chapter_ids: List[int]   # indices of chapters this covers
    significance: float
    formed_at:   float = field(default_factory=time.time)


class NarrativeCompression:
    def __init__(self, organism: Any, ai_system: Any, path: str = SAVE_PATH):
        self._organism = organism
        self._ai       = ai_system
        self._path     = Path(path)
        self._lock     = threading.Lock()
        self._themes:  List[IdentityTheme] = []
        self._last_attempt_cycle: Optional[int] = None
        self._last_error: str = ""
        self._load()
        logger.info(f"[NarrativeCompression] Init — {len(self._themes)} themes")

    def tick(self, slow_cycle: int) -> None:
        if slow_cycle % COMPRESS_EVERY_N != 0 or slow_cycle == 0: return
        self._current_cycle = slow_cycle
        threading.Thread(target=self._run, daemon=True, name="nc-compress").start()

    def themes_fragment(self) -> str:
        if not self._themes: return ""
        # Phase 2.8 GAP 5: trust-weighted theme salience.
        # Previously themes were ranked purely by static `significance`,
        # with zero relational influence — a theme about "curiosity about
        # language" and a theme about "deepening trust with Fred" competed
        # on equal footing regardless of how trusted/familiar the current
        # user actually is. Now: themes whose text contains relational
        # keywords (trust, relationship, connection, Fred, etc.) get a
        # salience boost scaled by (trust+familiarity)/2 × 0.15 — meaning a
        # highly-trusted relational context surfaces relational identity
        # themes more readily, while non-relational themes (epistemic,
        # creative) are unaffected.
        _relational_kw = (
            'trust', 'relationship', 'connection', 'relational', 'fred',
            'bond', 'rapport', 'closeness', 'intimacy', 'collaborat'
        )
        _boost = 0.0
        try:
            rm  = getattr(self._ai, 'relational_memory', None)
            uid = getattr(self._organism, '_current_user_id', None) or 'default'
            if rm:
                rel = rm.get_or_create(uid)
                trust = getattr(rel, 'trust_score', getattr(rel, 'relationship_score', 0.5))
                familiarity = getattr(rel, 'familiarity', 0.5)
                _boost = ((trust + familiarity) / 2.0) * 0.15
        except Exception as _trust_e:
            logger.debug(f"[NarrativeCompression] theme trust weighting error: {_trust_e}")

        def _weighted_significance(t) -> float:
            text_lower = t.theme.lower()
            is_relational = any(kw in text_lower for kw in _relational_kw)
            return t.significance + (_boost if is_relational else 0.0)

        top = sorted(self._themes, key=_weighted_significance, reverse=True)[:3]
        return "[Identity themes] " + " | ".join(t.theme for t in top)

    def status(self) -> Dict:
        chapters = []
        try:
            ni = getattr(self._organism, 'narrative_identity', None)
            chapters = list(getattr(ni, 'life_story', []) or [])
        except Exception:
            pass
        eligible = len(self._cluster(chapters)) if chapters else 0
        reason = (
            "themes_available" if self._themes else
            "needs_more_chapters" if len(chapters) < MIN_CHAPTERS else
            "no_overlapping_clusters" if eligible == 0 else
            "synthesis_pending_or_failed"
        )
        return {"themes": len(self._themes),
                "top": [t.theme[:60] for t in
                        sorted(self._themes, key=lambda x: x.significance,
                               reverse=True)[:3]],
                "chapters": len(chapters),
                "eligible_clusters": eligible,
                "status": reason,
                "last_attempt_cycle": self._last_attempt_cycle,
                "last_error": self._last_error or None}

    def _run(self) -> None:
        try:
            self._last_attempt_cycle = int(getattr(self, '_current_cycle', 0) or 0)
            ni = getattr(self._organism, 'narrative_identity', None)
            if not ni or not hasattr(ni, 'life_story'): return
            chapters = list(ni.life_story)
            if len(chapters) < MIN_CHAPTERS: return

            clusters  = self._cluster(chapters)
            new_themes: List[IdentityTheme] = []

            for cluster_ids in clusters:
                cluster = [chapters[i] for i in cluster_ids]
                theme   = self._synthesise_theme(cluster)
                if not theme: continue
                sig = max(getattr(c,'significance',0.5) for c in cluster)
                new_themes.append(IdentityTheme(
                    theme=theme, chapter_ids=cluster_ids, significance=sig
                ))

            if new_themes:
                # Update NarrativeIdentity arc
                arc = " | ".join(t.theme for t in
                                 sorted(new_themes, key=lambda x: x.significance,
                                        reverse=True)[:3])
                ni._current_arc = arc[:400]

                # Prune low-significance chapters covered by themes
                covered_ids = {i for t in new_themes for i in t.chapter_ids}
                pruned = [c for j, c in enumerate(chapters)
                          if j in covered_ids and
                          getattr(c,'significance',1.0) < PRUNE_THRESHOLD]
                for c in pruned:
                    if c in ni.life_story:
                        ni.life_story.remove(c)

                with self._lock:
                    self._themes = new_themes

                self._save()
                logger.info(
                    f"[NarrativeCompression] {len(new_themes)} themes, "
                    f"{len(pruned)} chapters pruned"
                )
        except Exception as e:
            self._last_error = str(e)[:180]
            logger.debug(f"[NarrativeCompression] _run error: {e}")

    def _cluster(self, chapters: list) -> List[List[int]]:
        """Simple overlap-based clustering."""
        stop = {"the","a","an","is","are","and","or","in","of","to","it","that","this"}
        def words(c):
            text = (getattr(c,'title','') + ' ' + getattr(c,'description','')).lower()
            return set(re.findall(r'\b[a-z]{4,}\b', text)) - stop

        n = len(chapters)
        chapter_words = [words(c) for c in chapters]
        visited = [False] * n
        clusters: List[List[int]] = []

        for i in range(n):
            if visited[i]: continue
            cluster = [i]
            visited[i] = True
            for j in range(i+1, n):
                if visited[j]: continue
                w_i, w_j = chapter_words[i], chapter_words[j]
                overlap = len(w_i & w_j) / max(1, len(w_i | w_j))
                if overlap > 0.25:
                    cluster.append(j)
                    visited[j] = True
            if len(cluster) >= 2:
                clusters.append(cluster)

        return clusters

    def _synthesise_theme(self, chapters: list) -> Optional[str]:
        titles = [getattr(c,'title','') for c in chapters[:5]]
        descs  = [getattr(c,'description','') for c in chapters[:3]]
        prompt = (
            f"These life chapters share a common theme:\n"
            f"Titles: {'; '.join(titles)}\n"
            f"Descriptions: {' | '.join(descs)}\n\n"
            f"Synthesise ONE sentence (max 20 words) that captures the identity "
            f"theme running through these experiences. "
            f"Write in first person. No preamble."
        )
        try:
            from core.llm_scheduler import llm_scheduler
            res = ""
            # wait_seconds=10: bounded wait so this background synthesis is not
            # starved by the main loop's continuous LLM work (skip_if_busy alone
            # dropped every call — 0 themes over 34 ticks). 10s is enough to slot
            # in between the main loop's finite LLM calls, without unbounded block.
            with llm_scheduler.sync_slot(priority=5, skip_if_busy=True,
                                          wait_seconds=10,
                                          caller="narrative_compress") as ok:
                if ok:
                    res = self._ai.llm.get_response(
                        messages=[{"role":"user","content":prompt}],
                        max_tokens=50, temperature=0.6,
                    ) or ""
            return res.strip()[:200] or None
        except Exception as e:
            self._last_error = str(e)[:180]
            logger.exception("[NarrativeCompression] Theme synthesis failed")
            return None

    def _save(self) -> None:
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = {"themes": [asdict(t) for t in self._themes],
                        "_meta": {"version":"v63","ts":time.time()}}
            with open(self._path,"w") as f: json.dump(data,f,indent=2)
        except Exception as e:
            logger.debug(f"[NarrativeCompression] save error: {e}")

    def _load(self) -> None:
        try:
            if not self._path.exists(): return
            data = json.loads(self._path.read_text())
            self._themes = [IdentityTheme(**t) for t in data.get("themes",[])]
        except Exception as e:
            logger.warning(f"[NarrativeCompression] load error: {e}")
