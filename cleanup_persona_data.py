"""
One-time cleanup script for PandoraBOX persona data.
Run from the lumina_v32/ directory:
    python cleanup_persona_data.py

Removes:
  - semantic_hub_* beliefs from identity.json (graph stats, not self-beliefs)
  - Noise/meta-process topics from curiosity.json (statement, question, etc.)
"""

import json
import shutil
from pathlib import Path
from datetime import datetime

PERSONA = Path("data/persona")

def backup(path: Path):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = path.with_suffix(f".{ts}.bak")
    shutil.copy2(path, bak)
    print(f"  Backup: {bak.name}")

NOISE_TOPICS = {
    "statement", "question", "task_oriented", "thought creating",
    "currently dimly", "recall middle", "chase", "middle", "actually",
    "recall yesterday", "dimly", "talking", "dive", "thought",
    "general", "unknown", "action", "goal action", "memory", "insight",
    "reflection", "curiosity", "emotion", "tension", "context", "content",
    "internal", "external", "active", "current", "recent", "cycle", "loop",
    "exploring", "understanding", "exploration", "continue", "verbe",
    "phrase", "topic", "concept", "word", "term", "type", "kind",
    "message", "prompt", "output", "result", "tick",
}

# ── identity.json ─────────────────────────────────────────────────────────────
identity_path = PERSONA / "identity.json"
if identity_path.exists():
    backup(identity_path)
    d = json.loads(identity_path.read_text())
    beliefs = d.get("beliefs", {})
    before = len(beliefs)
    d["beliefs"] = {k: v for k, v in beliefs.items()
                    if not k.startswith("semantic_hub_")}
    removed = before - len(d["beliefs"])
    identity_path.write_text(json.dumps(d, indent=2, ensure_ascii=False))
    print(f"identity.json: removed {removed} semantic_hub_* beliefs ({before} → {len(d['beliefs'])})")
else:
    print("identity.json not found, skipping")

# ── curiosity.json ────────────────────────────────────────────────────────────
curiosity_path = PERSONA / "curiosity.json"
if curiosity_path.exists():
    backup(curiosity_path)
    c = json.loads(curiosity_path.read_text())
    topics = c.get("topics", {})
    before_t = len(topics)
    c["topics"] = {k: v for k, v in topics.items()
                   if k not in NOISE_TOPICS and len(k) > 3}
    removed_t = before_t - len(c["topics"])
    curiosity_path.write_text(json.dumps(c, indent=2, ensure_ascii=False))
    print(f"curiosity.json: removed {removed_t} noise topics ({before_t} → {len(c['topics'])})")
    
    # Show top topics after cleanup
    sorted_t = sorted(c["topics"].items(),
                      key=lambda x: x[1].get("curiosity", 0) if isinstance(x[1], dict) else 0,
                      reverse=True)[:6]
    print("  Top topics after cleanup:")
    for name, data in sorted_t:
        level = data.get("curiosity", 0) if isinstance(data, dict) else data
        print(f"    {name!r}: {level:.3f}")
else:
    print("curiosity.json not found, skipping")

print("\nCleanup complete. Restart PandoraBOX to apply changes.")
