import { lazy, Suspense, useEffect, useState, type CSSProperties } from 'react';
import { ArrowUpRight, Pause, Play, RotateCcw, Workflow } from 'lucide-react';
import './cognitive-observatory.css';

const Sculpture = lazy(() => import('./CognitiveSculpture'));
const colors = ['#72ead4', '#ffac86', '#92b9ff', '#e5c678', '#e9eff2', '#b6a1ef'];
const record = (v: unknown): Record<string, any> => v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, any> : {};
const number = (v: unknown): number | null => typeof v === 'number' && Number.isFinite(v) ? v : null;
const unit = (v: unknown) => { const n = number(v); return n === null ? null : Math.max(0, Math.min(1, n)); };
const words = (v: unknown, empty = 'Awaiting data') => typeof v === 'string' && v ? v.replaceAll('_', ' ') : empty;
const percent = (v: number | null) => v === null ? '--' : `${Math.round(v * 100)}%`;
const duration = (v: unknown) => { const n = number(v); const seconds = Math.round(Math.max(0, n ?? 0)); return n === null ? '--' : `${Math.floor(seconds / 60)}m ${seconds % 60}s`; };

export default function CognitiveObservatory({ data, mode, connected, reduced, personaName = 'Lumina' }: { data: Record<string, unknown> | null; mode: string; connected: boolean; reduced: boolean; personaName?: string }) {
  const [selected, setSelected] = useState(0);
  const [paused, setPaused] = useState(false);
  const [reset, setReset] = useState(0);
  const [now, setNow] = useState(Date.now());
  useEffect(() => { const timer = setInterval(() => setNow(Date.now()), 2000); return () => clearInterval(timer); }, []);
  const timestamp = number(data?.fetched_at);
  const fresh = timestamp !== null && now - (timestamp > 1e12 ? timestamp : timestamp * 1000) < 25000;
  const live = connected && fresh;
  const workspace = record(data?.global_workspace), attention = record(data?.attention), motivation = record(data?.mf);
  const resources = record(data?.ec), planner = record(data?.long_horizon), orchestrator = record(data?.orchestrator);
  const dominantPlan = record(planner.dominant_plan);
  const clock = record(orchestrator.clock), drives = record(orchestrator.drives);
  const available = orchestrator.available === true;
  const running = available && orchestrator.running === true;
  const plans = Array.isArray(planner.plans) ? planner.plans : [];
  const currentPlan = plans.find((p: any) => number(p.completed) !== null && number(p.steps) !== null && p.completed < p.steps);
  const progress = currentPlan && currentPlan.steps > 0 ? unit(currentPlan.completed / currentPlan.steps) : null;
  // A fresh plan is active even when no step has completed yet. Keep the
  // factual 0/4 progress in the detail text, but give the sculpture a small
  // live planning signal so "planning exists" is not rendered as "nothing".
  const planningActivity = currentPlan
    ? Math.max(progress ?? 0, 0.14 + (unit(currentPlan.reasoning_confidence) ?? 0) * 0.10)
    : null;
  const reasoningConfidence = unit(dominantPlan.reasoning_confidence);
  const networks = [
    { name: 'Attention', value: words(attention.primary, 'No active focus'), metric: unit(record(attention.weights)[attention.primary]), label: 'Attention allocation', detail: words(attention.workspace_topic, 'No workspace topic recorded') },
    { name: 'Motivation', value: words(motivation.dominant, 'No dominant drive'), metric: unit(record(motivation.drive_vector)[motivation.dominant]), label: 'Dominant drive intensity', detail: `${number(motivation.needs_detected) ?? '--'} needs detected` },
    { name: 'Workspace', value: words(workspace.focus, 'No active focus'), metric: unit(workspace.confidence), label: 'Workspace confidence', detail: `${Array.isArray(workspace.active_hypotheses) ? workspace.active_hypotheses.length : '--'} active hypotheses` },
    { name: 'Planning', value: currentPlan?.objective || 'No active plan recorded', metric: planningActivity, label: 'Active planning signal', detail: currentPlan ? `Progress ${currentPlan.completed} of ${currentPlan.steps} steps · next ${words(currentPlan.next)}` : 'Awaiting a recorded plan' },
    { name: 'Energy', value: resources.in_dream_mode ? 'Dream mode' : 'Cognitive reserve', metric: unit(resources.cognitive_energy), label: 'Cognitive energy', detail: `Attention reserve ${percent(unit(resources.attention))}` },
    { name: 'Orchestrator', value: available ? words(orchestrator.last_activity) : 'Telemetry unavailable', metric: unit(drives.coherence), label: 'Drive coherence', detail: available ? `${number(orchestrator.queued_events) ?? '--'} queued events / cycle ${number(orchestrator.cycle_count) ?? '--'}` : 'Awaiting Orchestrator telemetry' },
  ];
  const chosen = networks[selected];
  const sculptureLevels = networks.map((network, index) => index === 2 ? reasoningConfidence : network.metric);
  const state = !live ? data ? 'Last snapshot' : 'Connecting' : mode === 'processing' ? 'Thinking' : mode === 'responding' ? 'Responding' : mode === 'listening' ? 'Listening' : 'Present';
  return <section className={`observatory ${live ? 'is-live' : ''}`} aria-label="Live cognitive activity" style={{ '--signal': colors[selected] } as CSSProperties}>
    <header className="observatory-heading"><div><span className="observatory-overline">COGNITIVE OBSERVATORY / 01</span><h2>{personaName}<span>In motion.</span></h2></div><div className="observatory-live"><i/>{state}<span>Cycle {number(data?.slow_cycle) ?? '--'}</span></div></header>
    <div className="observatory-stage">
      <div className="observatory-art">
        <Suspense fallback={<div className="sculpture-fallback">Loading cognitive sculpture...</div>}><Sculpture selected={selected} onSelect={setSelected} levels={sculptureLevels} workspace={workspace} moving={!paused && !reduced && live} live={live} reset={reset}/></Suspense>
        <div className="sculpture-caption"><span>0{selected + 1} / {chosen.name}</span><strong>{chosen.label}</strong><b>{percent(chosen.metric)}</b></div>
        <div className="sculpture-controls"><button aria-label={paused ? 'Resume sculpture motion' : 'Pause sculpture motion'} title={paused ? 'Resume motion' : 'Pause motion'} aria-pressed={paused} onClick={() => setPaused(p => !p)}>{paused ? <Play size={16}/> : <Pause size={16}/>}</button><button aria-label="Reset sculpture view" title="Reset view" onClick={() => setReset(r => r + 1)}><RotateCcw size={16}/></button></div>
      </div>
      <aside className="observatory-networks" aria-label="Cognitive networks"><div className="network-heading"><span>FUNCTIONAL NETWORKS</span><span>06</span></div>
        <div className="network-selector">{networks.map((network, i) => <button key={network.name} className={selected === i ? 'selected' : ''} style={{ '--network-color': colors[i] } as CSSProperties} onClick={() => setSelected(i)} aria-pressed={selected === i}><span className="network-index">0{i + 1}</span><i/><span>{network.name}</span><b>{percent(network.metric)}</b></button>)}</div>
        <div className="network-insight" aria-live="polite"><span>{chosen.label}</span><h3>{chosen.value}</h3><p>{chosen.detail}</p><div className="network-scale"><i style={{ width: `${(chosen.metric ?? 0) * 100}%` }}/></div></div>
      </aside>
    </div>
    <section className="orchestration-console" aria-label="Orchestrator activity"><div className="orchestration-current"><span className="console-kicker"><Workflow size={15}/> ORCHESTRATOR <i className={live && running ? 'running' : ''}/></span><h3>{!available ? 'Awaiting connection' : !live ? 'Last recorded activity' : running ? words(orchestrator.last_activity) : 'Stopped'}</h3><span>{available ? `${number(orchestrator.queued_events) ?? '--'} queued events` : 'Telemetry unavailable'}<a href="/orchestrator" target="_blank" rel="noreferrer" title="Open full Orchestrator controls" aria-label="Open full Orchestrator controls"><ArrowUpRight size={15}/></a></span></div>
      <div className="orchestration-clock"><span>REFLECTION</span><strong>{duration(clock.medium_in_sec)}</strong><small>{live && running ? 'Until next cycle' : 'Recorded countdown'}</small></div>
      <div className="orchestration-clock"><span>CONSOLIDATION</span><strong>{duration(clock.slow_in_sec)}</strong><small>{live && running ? 'Until next cycle' : 'Recorded countdown'}</small></div>
      <div className="orchestration-clock"><span>EVOLUTION</span><strong>{duration(clock.evolution_in_sec)}</strong><small>{live && running ? 'Until next cycle' : 'Recorded countdown'}</small></div>
      <div className="orchestration-drives"><span>DRIVE VECTOR</span><div>{['curiosity', 'coherence', 'energy', 'social', 'goal_progress', 'homeostasis'].map((key, i) => <div key={key} title={`${words(key)}: ${percent(unit(drives[key]))}`}><i style={{ height: `${(unit(drives[key]) ?? 0) * 100}%`, background: colors[i] }}/><small>{['CU', 'CO', 'EN', 'SO', 'GO', 'HO'][i]}</small></div>)}</div></div>
    </section>
    <footer className="observatory-foot"><span>{live ? 'LIVE SUBSYSTEM STATE' : 'AWAITING FRESH TELEMETRY'}</span><span>{timestamp ? new Date(timestamp > 1e12 ? timestamp : timestamp * 1000).toLocaleTimeString() : '--'} / {personaName.toUpperCase()}</span></footer>
  </section>;
}
