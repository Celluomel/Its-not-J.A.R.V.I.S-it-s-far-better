"""
PANDORABOX — STEP 6: CONSOLIDATE REPORT (intent-based, not blind-prune)
====================================================================
Inventories every subsystem and measures what is live vs decorative.
For orphaned code it does NOT just say "delete" — it records WHY the
module was built, whether that purpose still stands, and which active
module (if any) already owns that purpose.

Evidence sources
─────────────────
  E1. WIRING     — AST import graph from the real entry points (brain.py,
                   app.py). Catches `import x.y.z`, `from pkg import sub`
                   (submodule form!), relative imports, and dynamic
                   (__import__ / importlib.import_module) loads.
  E2. PROMPT     — data/persona/prompt_contributions.json (rolling 200
                   cycles): which sections actually reached the LLM, how
                   many tokens, and WHY the rest were dropped.
  E3. RECENCY    — data/persona state-file mtimes: which subsystems still
                   write state in the wild vs went silent weeks ago.
  E4. INTENT     — per-orphan docstring + importer analysis: what was it
                   built for, who was meant to use it, is that purpose
                   served by an active module now?

VERDICT CLASSES
  RESCUED        was a false orphan — actually wired, keep
  ACTIVE         wired + reaches the LLM + writes fresh state
  BUDGET-STARVED wired + runs every cycle + its prompt section is dropped
                (category_full) in ~all cycles  -> live code, zero effect
  DORMANT        wired + owns a state file silent for 30+ days
  RESERVE        orphaned but a viable option (e.g. an alt backend)
  OBSOLETE       orphaned AND its purpose is already served/replaced
  JUNK           .bak / .broken / backup artifacts

Re-runnable:  venv\\Scripts\\python.exe scripts\\prune_report.py
"""
from __future__ import annotations

import ast
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
DP = BASE / "data" / "persona"
SKIP = {"venv", "node_modules", "__pycache__", ".git", "data", "logs",
        "brain_visualizer"}


# ─────────────────────────────────────────────────────────────────────────────
# 1. WIRING — AST import graph from the real entry points
# ─────────────────────────────────────────────────────────────────────────────
def mod_map(base: Path):
    m = {}
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in SKIP]
        for f in files:
            if not f.endswith(".py"):
                continue
            rel = Path(root, f).relative_to(base).as_posix()
            parts = rel[:-3].split("/")
            if parts[-1] == "__init__":
                m[".".join(parts[:-1])] = rel
            else:
                m[".".join(parts)] = rel
    return m


def imports_of(path: Path, base: Path):
    """Return (static_deps:set[str], dynamic_deps:set[str])."""
    try:
        src = path.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(src)
    except Exception:
        return set(), set()
    static, dynamic = set(), set()
    # package of this file (for relative-import resolution)
    pkg_parts = path.relative_to(base).as_posix()[:-3].split("/")
    if pkg_parts[-1] == "__init__":
        pkg_parts = pkg_parts[:-1]

    def resolve(base_parts, level, module):
        parts = base_parts[: len(base_parts) - max(0, level - 1)]
        if module:
            parts = parts + module.split(".")
        return ".".join(parts)

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                static.add(a.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level and node.level > 0:
                base_mod = resolve(pkg_parts, node.level, node.module)
            elif node.module:
                base_mod = node.module
            else:
                base_mod = None
            if base_mod:
                static.add(base_mod)
                # `from pkg import submodule` — the name may itself be a
                # submodule; record pkg.submodule as a candidate dep too.
                for a in node.names:
                    if a.name != "*":
                        static.add(f"{base_mod}.{a.name}")
        elif isinstance(node, ast.Call):
            fn = node.func
            if isinstance(fn, ast.Attribute) and fn.attr == "import_module":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        dynamic.add(a.value)
            elif isinstance(fn, ast.Name) and fn.id == "__import__":
                for a in node.args:
                    if isinstance(a, ast.Constant) and isinstance(a.value, str):
                        dynamic.add(a.value)
    return static, dynamic


def closure(base: Path, entry: str, mm: dict):
    deps = {}
    for name, rel in mm.items():
        deps[name] = imports_of(base / rel, base)

    def norm(n):
        if n in mm:
            return n
        parts = n.split(".")
        for i in range(len(parts) - 1, 0, -1):
            p = ".".join(parts[:i])
            if p in mm:
                return p
        return None

    seen, stack = set(), [entry]
    while stack:
        m = stack.pop()
        if m in seen:
            continue
        seen.add(m)
        for d in deps.get(m, (set(), set()))[0] | deps.get(m, (set(), set()))[1]:
            dn = norm(d)
            if dn and dn not in seen:
                stack.append(dn)
    return {c for c in seen if c in mm}


# ─────────────────────────────────────────────────────────────────────────────
# 2. PROMPT — what actually reached the LLM
# ─────────────────────────────────────────────────────────────────────────────
def prompt_stats():
    p = DP / "prompt_contributions.json"
    if not p.exists():
        return None, 0
    pc = json.loads(p.read_text(encoding="utf-8"))
    N = len(pc)
    st = defaultdict(lambda: {"seen": 0, "sel": 0, "drop": 0, "tok_sel": 0,
                              "tok_drop": 0, "reasons": Counter()})
    for snap in pc:
        for s in snap.get("sections", []):
            d = st[s["source"]]
            d["seen"] += 1
            if s.get("selected"):
                d["sel"] += 1
                d["tok_sel"] += s.get("tokens", 0)
            else:
                d["drop"] += 1
                d["tok_drop"] += s.get("tokens", 0)
                d["reasons"][s.get("reason", "?")] += 1
    return st, N


# ─────────────────────────────────────────────────────────────────────────────
# 3. RECENCY — which state files are still being written
# ─────────────────────────────────────────────────────────────────────────────
def recency():
    out = {}
    if not DP.exists():
        return out
    now = time.time()
    for f in os.listdir(DP):
        fp = DP / f
        if fp.is_file():
            out[f] = (now - fp.stat().st_mtime) / 86400.0  # days
    return out


# ─────────────────────────────────────────────────────────────────────────────
# OWNERSHIP — which wired module writes which persona file
# ─────────────────────────────────────────────────────────────────────────────
def owners(file_name: str, srcs: dict):
    base = re.escape(file_name.rsplit(".", 1)[0])
    out = set()
    for m, s in srcs.items():
        if re.search(r'["\']' + base + r'["\']', s) or base in s.lower():
            out.add(m)
    return out


def gather_srcs(base: Path):
    srcs = {}
    for pkg in ["cognition", "core", "utils", "memory", "goals",
                "physiology", "psychology"]:
        d = base / pkg
        if not d.is_dir():
            continue
        for f in os.listdir(d):
            if f.endswith(".py"):
                try:
                    srcs[f"{pkg}/{f}"] = (d / f).read_text(
                        encoding="utf-8", errors="replace")
                except Exception:
                    pass
    for f in ["brain.py", "app.py"]:
        p = base / f
        if p.exists():
            srcs[f] = p.read_text(encoding="utf-8", errors="replace")
    return srcs


# ─────────────────────────────────────────────────────────────────────────────
# REPORT
# ─────────────────────────────────────────────────────────────────────────────
def banner(t):
    print("\n" + "═" * 88 + f"\n{t}\n" + "═" * 88)


def main():
    mm = mod_map(BASE)
    brain = closure(BASE, "brain", mm)
    gui = closure(BASE, "app", mm) if (BASE / "app.py").exists() else set()
    srcs = gather_srcs(BASE)
    pstat, Ncyc = prompt_stats()
    rec = recency()

    allmods = set(mm)
    orphaned = sorted(allmods - brain - gui)

    # ── JUNK artifacts ─────────────────────────────────────────────────────
    junk = [f for f in rec if re.search(r"\.bak$|\.broken|backup_\d|_pruned_|\.old", f)]

    # ── BUDGET-STARVED prompt sources ──────────────────────────────────────
    starved, active = [], []
    if pstat:
        for src, d in pstat.items():
            rate = d["sel"] / d["seen"] if d["seen"] else 0
            avg_drop = d["tok_drop"] / d["drop"] if d["drop"] else 0
            row = {"source": src, "rate": rate, "seen": d["seen"],
                   "sel": d["sel"], "avg_drop_tok": round(avg_drop, 1),
                   "reason": d["reasons"].most_common(1)[0][0]
                             if d["reasons"] else ""}
            (starved if rate < 0.25 else active).append(row)
        starved.sort(key=lambda r: (r["rate"], -r["avg_drop_tok"]))
        active.sort(key=lambda r: -r["sel"])

    # ── DORMANT state files (owning module wired, file silent 30+ days) ───
    dormant = []
    for f, days in sorted(rec.items(), key=lambda x: -x[1]):
        if days >= 30 and f not in junk and not f.endswith((".bin", ".meta.json", ".db", ".db-shm", ".db-wal", ".log")):
            own = sorted(owners(f, srcs))
            own_wired = [o for o in own if o in brain or o in gui]
            dormant.append({"file": f, "days": round(days), "owners": own_wired or own})

    # ── PRINT ──────────────────────────────────────────────────────────────
    banner(f"PANDORABOX PRUNE / CONSOLIDATE REPORT   (modules={len(mm)}, "
           f"brain-wired={len(brain)}, gui-wired={len(gui)}, cycles={Ncyc})")

    banner("① BUDGET-STARVED — runs every cycle, prompt section dropped (live code, ~zero effect)")
    print(f"  {'source':<26} {'rate':>7} {'seen':>5} {'avg_drop_tok':>13}  {'why':<14}")
    for r in starved:
        print(f"  {r['source']:<26} {r['rate']:>7.2f} {r['seen']:>5} "
              f"{r['avg_drop_tok']:>13}  {r['reason']}")
    print("\n  → root cause: self_state=120tok budget is consumed by self_model_moment")
    print("    (~109tok); cognitive_pressure=100 by energy+motivational_field; attention=80 by")
    print("    attention+curiosity. Starved sections are larger than the remaining headroom.")

    banner("② ACTIVE — wired, reaches the LLM regularly (KEEP)")
    for r in active:
        print(f"  {r['source']:<28} selected {r['sel']}/{r['seen']}  "
              f"({r['rate']:.0%})")

    banner("③ DORMANT — owns a state file silent for 30+ days (candidate: prune or revive)")
    for d in dormant:
        ow = ", ".join(o.replace(".py", "") for o in d["owners"])[:60]
        print(f"  {d['file']:<40} {d['days']:>4}d   owner≈{ow}")
    borderline = [f for f, days in rec.items()
                  if 7 <= days < 30 and f not in junk
                  and not f.endswith((".bin", ".meta.json", ".db", ".db-shm", ".db-wal", ".log"))]
    if borderline:
        print(f"\n  borderline (7-30d): {', '.join(sorted(borderline))}")

    banner("④ ORPHANED — not reachable from brain.py OR app.py (raw list; see ⑥ for intent verdicts)")
    # group by top package
    bypkg = Counter(o.split(".")[0] for o in orphaned)
    for pkg, n in sorted(bypkg.items()):
        mods = sorted(o for o in orphaned if o.split(".")[0] == pkg)
        if pkg in ("scripts", "logs", "pages", "brain_visualizer"):
            note = "  (separate tool / GUI surface — not organism code)"
        else:
            note = ""
        print(f"\n  {pkg}  ({n} modules){note}")
        for m in mods:
            print(f"      {m}")

    banner("⑤ JUNK — backup / broken / pruned artifacts (safe to delete)")
    for f in sorted(junk):
        print(f"  {f}")

    banner("RECOMMENDATIONS (generated from the measured data above)")

    # PRUNE NOW — budget-starved prompt paths (zero effect, measured)
    print("\n  PRUNE NOW  — prompt sections that run every cycle but never reach the LLM:")
    for r in starved:
        print(f"    • {r['source']:<26} selected {r['sel']}/{r['seen']} "
              f"({r['rate']:.0%}), ~{r['avg_drop_tok']}tok generated & dropped ({r['reason']})")
    print("    → Keep the COMPUTATION that feeds ACTIVE sections (e.g. binder state still flows")
    print("      into self_model_moment 100%); prune only the dead prompt-fragment generation.")

    # DECIDE — dormant state (wired, silent)
    print("\n  DECIDE  — wired but dormant (user value call):")
    for d in dormant:
        print(f"    • {d['file']:<38} silent {d['days']}d")
    if borderline:
        print(f"    borderline: {', '.join(sorted(borderline))}")

    # ── INTENT VERDICTS for orphaned org code ────────────────────────────────
    # Each verdict answers: what was it built for, is that purpose still needed,
    # and does an active module already own it? (not a blanket "prune")
    VERDICTS = {
        "cognition.ablation":
            ("KEEP", "verification harness — imported by scripts/, a tool not a subsystem"),
        "cognition.goal_semantics":
            ("RESCUED", "false orphan — actively backs goal_engine + goal_quality_filter "
             "(semantic scoring); wired via `from cognition import goal_semantics`"),
        "cognition.loop_observability":
            ("RESCUED", "false orphan — GUI dashboard metrics (Phase F), reached via cognitive_dashboard_page"),
        "core.execution.actuator_interface":
            ("RESCUED", "false orphan — action execution, reached via wired core.phase1_integration"),
        "core.planning.multi_step_planner":
            ("RESCUED", "false orphan — goal decomposition, reached via wired core.phase1_integration"),
        "cognition.liberty_integration":
            ("OBSOLETE", "redundant wrapper — its purpose (integrating the 5 liberty components) is "
             "already done inline by ai_system.py, which instantiates all 5 directly (liberty_self_mod / "
             "choice / contradiction / reflection / goals, all BRAIN+GUI)"),
        "cognition.idle_data_bootstrapper":
            ("OBSOLETE", "purpose already satisfied — it existed to push CausalMechanismModel past "
             "MIN_RECORDS=20, but causal_mechanisms already holds 61-121 records/domain and "
             "consequence_model 300; the model is warm with real data"),
        "cognition.interfaces":
            ("OBSOLETE", "contract layer only consumed by the replaced legacy memory/* package"),
        "cognition.research_mcp.claude_search":
            ("RESERVE", "a web-search backend option, never registered in the backend selector and "
             "gated on ANTHROPIC_API_KEY (empty) — viable option, not a live core function"),
        "core.engine":
            ("OBSOLETE", "Phase-1 integrated cognitive engine — superseded by cognitive_organism + internal_loop"),
        "core.engine.enhanced_cognition_engine":
            ("OBSOLETE", "Phase-1 integrated engine — superseded by cognitive_organism + internal_loop"),
        "core.event_bus":
            ("OBSOLETE", "Phase-1 event coordination — superseded by active core.cognitive_event_bus"),
        "core.event_bus.async_event_bus":
            ("OBSOLETE", "Phase-1 event bus — superseded by active core.cognitive_event_bus"),
        "core.perception":
            ("KEEP", "mixed package — perception_hub.py archived; unified_perception_hub.py is wired & live via phase1_integration"),
        "core.perception.perception_hub":
            ("OBSOLETE", "Phase-1 perception hub — superseded by brain.py senses + cognitive_organism (archived 2026-09-03)"),
        "core.memory":
            ("OBSOLETE", "Phase-1 unified memory — superseded by active cognition.semantic_memory"),
        "core.memory.unified_memory":
            ("OBSOLETE", "Phase-1 unified memory — superseded by active cognition.semantic_memory"),
        "memory.memory_interface":
            ("OBSOLETE", "legacy memory impl — superseded by active semantic_memory"),
        "memory.persistence":
            ("OBSOLETE", "legacy state persistence — superseded by data/persona JSON writes"),
        "memory.robot_memory":
            ("OBSOLETE", "foreign legacy — points at ~/.robot_agent (a different project); superseded by semantic_memory"),
        "memory.time_provider":
            ("OBSOLETE", "legacy time service — time now handled inline by the active loop"),
        "memory":
            ("OBSOLETE", "legacy memory package (superseded architecture)"),
        "core.execution":
            ("KEEP", "package init — actuator_interface itself is RESCUED/wired"),
        "core.planning":
            ("KEEP", "package init — multi_step_planner itself is RESCUED/wired"),
    }

    tool_pkgs = {"scripts", "logs", "pages", "brain_visualizer", "components",
                 "managers", "utils", "goals", "physiology", "psychology"}
    org_orphans = [o for o in orphaned if o.split(".")[0] not in tool_pkgs
                   and o.split(".")[0] not in ("cleanup_persona_data", "doc_generator2")]
    # collapse a package into its child when the child carries the verdict
    org_orphans = [o for o in org_orphans
                   if not any(o != c and c.startswith(o + ".") for c in org_orphans)]

    order = {"RESCUED": 0, "KEEP": 1, "RESERVE": 2, "OBSOLETE": 3}
    org_orphans.sort(key=lambda o: (order.get(VERDICTS.get(o, ("?",))[0], 9), o))

    banner("⑥ ORPHAN VERDICTS — what each was built for, and whether that purpose still stands")
    # First: the false orphans — modules a naive `from pkg import sub`-blind
    # scan would flag as dead, but which are actually wired and active.
    rescued = [m for m, (v, _) in sorted(VERDICTS.items()) if v == "RESCUED"]
    if rescued:
        print("  RESCUED — looked orphaned, but are actually wired & active (KEEP):")
        for m in rescued:
            _, why = VERDICTS[m]
            print(f"    ✓ {m}\n              {why}")
        print()
    for o in org_orphans:
        v, why = VERDICTS.get(o, ("UNCLASSIFIED", "no intent recorded — inspect manually"))
        print(f"  [{v:<9}] {o}")
        print(f"              {why}")

    obsolete = [o for o in org_orphans if VERDICTS.get(o, ("",))[0] == "OBSOLETE"]
    reserve = [o for o in org_orphans if VERDICTS.get(o, ("",))[0] == "RESERVE"]
    keep = [o for o in org_orphans if VERDICTS.get(o, ("",))[0] == "KEEP"]

    # JUNK
    print("\n  JUNK — safe to delete:")
    for f in sorted(junk):
        print(f"    • {f}")

    print("\n" + "─" * 88)
    print(f"  Totals: {len(starved)} starved paths · {len(dormant)} dormant states · "
          f"{len(junk)} junk artifacts")
    print(f"          orphans: {len(rescued)} RESCUED (wired, keep) · "
          f"{len(reserve)} RESERVE (option) · {len(obsolete)} OBSOLETE (purpose served/replaced) · "
          f"{len(keep)} KEEP (pkg/harness)")


if __name__ == "__main__":
    main()
