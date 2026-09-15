"""
LUMINA V32 — Phase 3 : Data Migrator
=====================================
One-time migration of existing persona data to canonical Phase 3 format.

Run once on deploy:
    python -m core.data.migrator

Or call from Python:
    from core.data.migrator import DataMigrator
    m = DataMigrator()
    report = m.run()
"""

import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from core.data.schemas import (
    normalize_goal, normalize_thought, normalize_thread,
    make_belief, make_goals_envelope, make_identity_envelope,
    TTL_MEDIUM, TTL_LONG, now_iso, now_ts, new_id,
)

logger = logging.getLogger(__name__)


class DataMigrator:
    """
    Migrates all key persona files to Phase 3 canonical format.

    Backs up everything first. Safe to re-run (idempotent).
    """

    def __init__(self, persona_dir: str = "data/persona"):
        self.persona_dir = Path(persona_dir)
        self.backup_dir = self.persona_dir / f"_backup_phase3_{int(time.time())}"
        self.report: Dict[str, Any] = {
            "started_at": now_iso(),
            "migrations": {},
            "errors": [],
        }

    # ── Entry point ───────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Run all migrations. Returns a report dict."""
        print("\n" + "="*60)
        print("  LUMINA PHASE 3 — DATA MIGRATION")
        print("="*60)

        # 1. Backup first
        self._backup()

        # 2. Migrate each file
        self._migrate_goals()
        self._migrate_thought_stream()
        self._migrate_thought_threads()
        self._migrate_tensions()
        self._create_identity()

        # 3. Summary
        self.report["finished_at"] = now_iso()
        total = sum(v.get("modified", 0) for v in self.report["migrations"].values())
        errors = len(self.report["errors"])

        print(f"\n✅ Migration complete — {total} records updated, {errors} errors")
        print(f"   Backup at: {self.backup_dir}")
        print("="*60 + "\n")

        return self.report

    # ── Backup ────────────────────────────────────────────────────────────────

    def _backup(self):
        """Copy all JSON files to backup directory."""
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        backed = 0
        for f in self.persona_dir.glob("*.json"):
            shutil.copy2(f, self.backup_dir / f.name)
            backed += 1
        print(f"  📦 Backed up {backed} files → {self.backup_dir.name}/")
        self.report["backup_dir"] = str(self.backup_dir)
        self.report["backed_up"] = backed

    # ── goals.json ────────────────────────────────────────────────────────────

    def _migrate_goals(self):
        path = self.persona_dir / "goals.json"
        if not path.exists():
            self._record("goals.json", 0, "file not found — skipped")
            return

        with open(path) as f:
            raw = json.load(f)

        # Normalize envelope
        if "goals" not in raw:
            raw = {"cycle_counter": 0, "goals": raw, "saved_at": now_iso()}

        goals_raw = raw["goals"]
        if isinstance(goals_raw, list):
            goals_raw = {g.get("id", new_id("goal_")): g for g in goals_raw}

        modified = 0
        for gid, goal in goals_raw.items():
            before = dict(goal)
            goal_n = normalize_goal(goal)
            goal_n["id"] = gid  # preserve existing id
            goals_raw[gid] = goal_n
            if goal_n != before:
                modified += 1

        raw["goals"] = goals_raw
        raw["saved_at"] = now_iso()

        self._save(path, raw)
        self._record("goals.json", modified, f"{len(goals_raw)} goals normalized")
        print(f"  ✅ goals.json        — {len(goals_raw)} goals, {modified} updated")

    # ── thought_stream.json ───────────────────────────────────────────────────

    def _migrate_thought_stream(self):
        path = self.persona_dir / "thought_stream.json"
        if not path.exists():
            self._record("thought_stream.json", 0, "file not found — skipped")
            return

        with open(path) as f:
            raw = json.load(f)

        if isinstance(raw, list):
            raw = {"thoughts": raw}
        if "thoughts" not in raw:
            raw = {"thoughts": []}

        thoughts = raw["thoughts"]
        modified = 0

        for i, t in enumerate(thoughts):
            before = dict(t)
            t_n = normalize_thought(t)

            # Assign TTL based on priority
            if "ttl" not in t:
                p = t.get("priority", 0.5)
                t_n["ttl"] = TTL_LONG if p >= 0.7 else TTL_MEDIUM

            # Compute expires_at if missing
            if "expires_at" not in t and t_n["ttl"] > 0:
                # Use timestamp + ttl as expiry
                try:
                    from datetime import datetime
                    ts = datetime.fromisoformat(
                        t.get("timestamp", now_iso()).replace("Z", "+00:00")
                    ).timestamp()
                    t_n["expires_at"] = ts + t_n["ttl"]
                except Exception:
                    t_n["expires_at"] = now_ts() + t_n["ttl"]

            thoughts[i] = t_n
            if t_n != before:
                modified += 1

        raw["thoughts"] = thoughts
        self._save(path, raw)
        self._record("thought_stream.json", modified, f"{len(thoughts)} thoughts normalized")
        print(f"  ✅ thought_stream    — {len(thoughts)} thoughts, {modified} updated")

    # ── thought_threads.json ──────────────────────────────────────────────────

    def _migrate_thought_threads(self):
        path = self.persona_dir / "thought_threads.json"
        if not path.exists():
            self._record("thought_threads.json", 0, "file not found — skipped")
            return

        with open(path) as f:
            raw = json.load(f)

        modified = 0
        for tid, thread in raw.items():
            if not isinstance(thread, dict):
                continue
            before = dict(thread)
            thread_n = normalize_thread(thread)
            thread_n["id"] = tid  # preserve key
            raw[tid] = thread_n
            if thread_n != before:
                modified += 1

        self._save(path, raw)
        self._record("thought_threads.json", modified, f"{len(raw)} threads normalized")
        print(f"  ✅ thought_threads   — {len(raw)} threads, {modified} updated")

    # ── tensions.json ─────────────────────────────────────────────────────────

    def _migrate_tensions(self):
        path = self.persona_dir / "tensions.json"
        if not path.exists():
            self._record("tensions.json", 0, "file not found — skipped")
            return

        with open(path) as f:
            raw = json.load(f)

        modified = 0
        if "history" not in raw:
            raw["history"] = []
            modified += 1
        if "last_updated" not in raw:
            raw["last_updated"] = now_ts()
            modified += 1

        self._save(path, raw)
        self._record("tensions.json", modified, "history + last_updated added")
        print(f"  ✅ tensions.json     — {modified} fields added")

    # ── identity.json (create if missing) ─────────────────────────────────────

    def _create_identity(self):
        path = self.persona_dir / "identity.json"

        if path.exists():
            with open(path) as f:
                raw = json.load(f)
            # Already exists — just ensure it has the right envelope
            if "beliefs" not in raw:
                raw = make_identity_envelope(raw)
                self._save(path, raw)
                self._record("identity.json", 1, "envelope added to existing file")
                print(f"  ✅ identity.json     — envelope added")
            else:
                self._record("identity.json", 0, "already exists — no changes")
                print(f"  ✅ identity.json     — already valid")
            return

        # Seed with core beliefs from self_concept or self_model if available
        seed_beliefs = {}

        for src_file in ["self_concept.json", "self_model.json"]:
            src = self.persona_dir / src_file
            if src.exists():
                try:
                    with open(src) as f:
                        concept = json.load(f)
                    for trait, val in concept.get("traits", {}).items():
                        bel = make_belief(
                            name=f"trait_{trait}",
                            value=float(val) if isinstance(val, (int, float)) else 0.5,
                            confidence=0.7,
                        )
                        seed_beliefs[bel["name"]] = bel
                    print(f"     └─ seeded {len(seed_beliefs)} beliefs from {src_file}")
                    break
                except Exception:
                    pass

        identity = make_identity_envelope(seed_beliefs)
        self._save(path, identity)
        self._record("identity.json", len(seed_beliefs), f"created with {len(seed_beliefs)} seed beliefs")
        print(f"  ✅ identity.json     — created ({len(seed_beliefs)} seed beliefs)")

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _save(self, path: Path, data: Any):
        import os
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        os.replace(tmp, path)

    def _record(self, filename: str, modified: int, note: str = ""):
        self.report["migrations"][filename] = {
            "modified": modified,
            "note": note,
        }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

    migrator = DataMigrator()
    report = migrator.run()

    import json as _json
    print("\nFull report:")
    print(_json.dumps(report, indent=2))
