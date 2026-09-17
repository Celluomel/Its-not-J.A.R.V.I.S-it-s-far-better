import { lazy, Suspense, useEffect, useRef, useState, type ReactNode } from 'react';
import { Activity, ArrowDown, ArrowLeft, ArrowUp, AudioLines, Check, ChevronRight, Copy, Database, Expand, Eye, History, Lightbulb, MessageCircle, Mic, MicOff, Network, PanelRightOpen, Pause, Phone, Play, PlugZap, RefreshCw, Rss, Search, Send, Settings, Sparkles, Square, ThumbsDown, ThumbsUp, Volume2, Wifi, Workflow, X } from 'lucide-react';
import Markdown from 'react-markdown';
import { appendReasoning, readEvents, visibleText } from './stream.mjs';
import { listenContinuously } from './audio';
import { formatHealthTimestamp } from './health-format.mjs';
import CognitiveActivity from './CognitiveObservatory';

type CognitiveTelemetry = {
  captured_at?: number;
  active_goals?: { id: string; topic: string; origin: string; priority: number; energy: number; activation: number }[];
  dominant_plan?: { objective: string; status: string; steps_completed: number; steps_total: number; completed_operations: string[]; next_operation?: string | null; reasoning_confidence: number; uncertainties: string[]; initial_decision?: string; decision_summary?: string; decision_revision?: number; decision_updated_at?: number; last_outcome?: { operation: string; success: boolean; result: string; observed_at: number } } | null;
};
type WebSource = { url: string; title: string; kind: 'page' | 'search_result' | 'fetch_failed'; status: string; detail?: string };
type Message = { id: string; role: 'user' | 'assistant'; text: string; stopped?: boolean; feedback?: 'positive' | 'negative'; telemetry?: CognitiveTelemetry; webSources?: WebSource[] };
type Status = { version: number; ready: boolean; busy: boolean; timestamp: number; uptime: number; cycle: number | null; engines: Record<string, boolean>; providers: Record<string, string>; response_language?: string; web_search_mode?: 'off' | 'auto' | 'always'; response_verbosity?: 'concise' | 'verbose'; emotional_state?: string; life_stage?: string; current_age?: number | null; presence_messages?: { id: string; text: string; timestamp: number }[]; timings: Record<string, number>; events: { id: string; label: string; timestamp: number; turn_id?: string }[] };
type NetworkState = { enabled: boolean; children: { id: string; name: string; url: string; role: string; online: boolean; latency_ms: number; turn_count: number }[] };
type VoiceMemSpaceData = { status?: { enabled?: boolean; available?: boolean; ready?: boolean; error?: string; observations?: number; retrievals?: number; consolidations?: number }; evaluation?: { observations?: number; retrievals?: number; nonempty_retrievals?: number; retrieval_hit_rate?: number | null }; observations?: { text?: string; timestamp?: number; status?: string }[]; retrievals?: { query?: string; result_count?: number; timestamp?: number }[]; cached?: { text?: string }[]; storage?: string };
type HomeAssistantDiscovery = { ok?: boolean; status?: string; count?: number; message?: string; entities?: { entity_id: string; state?: string; friendly_name?: string; unit?: string | null; domain?: string; device_class?: string | null; signal?: string | null }[] };
const languageOptions = [['auto', 'Auto'], ['en', 'EN'], ['uk', 'UK'], ['fr', 'FR'], ['es', 'ES'], ['de', 'DE'], ['it', 'IT'], ['pt', 'PT'], ['zh-cn', 'ZH'], ['ru', 'RU'], ['ar', 'AR'], ['ja', 'JA'], ['ko', 'KO'], ['hi', 'HI'], ['nl', 'NL'], ['pl', 'PL'], ['tr', 'TR'], ['sv', 'SV']] as const;
type DialogueMessage = { text: string; sender_id: string; sender_name: string; direction: 'out' | 'in'; timestamp: number };
const storageKey = 'lumina.interface.transcript.v1';
const telemetryVisibilityKey = 'lumina.interface.showTelemetry';
const LazyOrganism = lazy(() => import('./Organism'));
function Organism(props: { active: boolean; energy: number; reduced: boolean; mode?: 'idle' | 'listening' | 'processing' | 'responding'; onObjectSelect?: (id: string) => void }) {
  return <div className="organism-render"><Suspense fallback={<div className="organism graphics-fallback" aria-label="Loading organism">Lumina</div>}><LazyOrganism {...props}/></Suspense><ReasoningOverlay/><ProactiveOverlay/></div>;
}

function VoiceMemSpace({ data }: { data: VoiceMemSpaceData | null }) {
  const status = data?.status;
  const evaluation = data?.evaluation;
  const label = !status?.available ? 'Unavailable' : status.enabled ? (status.ready ? 'Ready' : 'Available') : 'Disabled';
  return <section className="voicemem-space" aria-label="VoiceMem memory space"><div className="voicemem-heading"><div><span className="eyebrow"><Database size={12}/> VOICEMEM / MEMORY SPACE</span><h2>Voice memory</h2></div><span className={`voicemem-state ${status?.ready ? 'ready' : ''}`}>{label}</span></div><div className="voicemem-metrics"><div><span>Observations</span><b>{evaluation?.observations ?? 0}</b></div><div><span>Retrievals</span><b>{evaluation?.retrievals ?? 0}</b></div><div><span>Consolidations</span><b>{status?.consolidations ?? 0}</b></div><div><span>Hit rate</span><b>{evaluation?.retrieval_hit_rate == null ? '—' : `${Math.round(evaluation.retrieval_hit_rate * 100)}%`}</b></div></div>{status?.error && <p className="voicemem-note">{status.error}</p>}{data?.cached?.length ? <div className="voicemem-cached"><span className="voicemem-label">Latest retrieved memories</span>{data.cached.slice(0, 3).map((item, index) => <p key={`${item.text}-${index}`}>{item.text}</p>)}</div> : data?.observations?.length ? <div className="voicemem-cached"><span className="voicemem-label">Latest local observations</span>{data.observations.slice(-3).reverse().map((item, index) => <p key={`${item.text}-${index}`}><span className="voicemem-observation-status">{item.status === 'fallback' ? 'local' : 'indexed'}</span>{item.text}</p>)}</div> : <p className="voicemem-empty">{status?.enabled ? 'No VoiceMem observations yet.' : 'Enable VoiceMem in Memory settings to record voice observations.'}</p>}</section>;
}

function ReasoningOverlay() {
  const [text, setText] = useState('');
  const [fading, setFading] = useState(false);
  const fadeTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const clearTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const hovering = useRef(false);
  const stopTimers = () => {
    if (fadeTimer.current) clearTimeout(fadeTimer.current);
    if (clearTimer.current) clearTimeout(clearTimer.current);
    fadeTimer.current = undefined;
    clearTimer.current = undefined;
  };
  const scheduleDismiss = () => {
    stopTimers();
    fadeTimer.current = setTimeout(() => setFading(true), 14000);
    clearTimer.current = setTimeout(() => setText(''), 17000);
  };
  const dismiss = () => {
    hovering.current = false;
    stopTimers();
    setFading(false);
    setText('');
  };
  useEffect(() => {
    const receive = (event: Event) => {
      const next = (event as CustomEvent<string>).detail || '';
      if (!next.trim()) return;
      setText(next);
      setFading(false);
      stopTimers();
      if (!hovering.current) scheduleDismiss();
    };
    window.addEventListener('lumina-reasoning', receive);
    return () => {
      window.removeEventListener('lumina-reasoning', receive);
      stopTimers();
    };
  }, []);
  return text ? <div className={`reasoning-popup ${fading ? 'is-fading' : ''}`} role="status" onMouseEnter={() => { hovering.current = true; stopTimers(); setFading(false); }} onMouseLeave={() => { hovering.current = false; scheduleDismiss(); }}><span className="reasoning-kicker"><Sparkles size={13}/>Reflection</span><button className="reasoning-close" type="button" aria-label="Close reflection" title="Close reflection" onClick={dismiss}><X size={14}/></button><p>{text}</p></div> : null;
}

function ProactiveOverlay() {
  const [message, setMessage] = useState<{ id: string; text: string } | null>(null);
  const dismissTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  useEffect(() => {
    const receive = (event: Event) => {
      const next = (event as CustomEvent<{ id: string; text: string }>).detail;
      if (!next?.text?.trim()) return;
      setMessage({ id: next.id, text: next.text.trim() });
      if (dismissTimer.current) clearTimeout(dismissTimer.current);
      dismissTimer.current = setTimeout(() => setMessage(current => current?.id === next.id ? null : current), 11000);
    };
    window.addEventListener('lumina-presence', receive);
    return () => { window.removeEventListener('lumina-presence', receive); if (dismissTimer.current) clearTimeout(dismissTimer.current); };
  }, []);
  return message ? <div key={message.id} className="proactive-presence" role="status" onMouseEnter={() => { if (dismissTimer.current) clearTimeout(dismissTimer.current); }} onMouseLeave={() => { dismissTimer.current = setTimeout(() => setMessage(current => current?.id === message.id ? null : current), 11000); }}><span className="proactive-kicker"><Eye size={12}/>Presence</span><p>{message.text}</p></div> : null;
}

function TelemetryTrace({ data }: { data: CognitiveTelemetry }) {
  const goals = data.active_goals || [];
  const plan = data.dominant_plan;
  return <aside className="message-telemetry" aria-label="Cognitive telemetry">
    <div className="message-telemetry-title"><Activity size={13}/><span>COGNITIVE TRACE</span>{data.captured_at && <time>{new Date(data.captured_at * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</time>}</div>
    <div className="message-telemetry-section"><span>ACTIVE GOALS</span>{goals.length ? <div className="telemetry-goals">{goals.map(goal => <div key={goal.id}><strong>{goal.topic}</strong><small>{goal.origin} · activation {goal.activation.toFixed(3)} · energy {goal.energy.toFixed(3)}</small></div>)}</div> : <p>No active goal recorded.</p>}</div>
    <div className="message-telemetry-section"><span>DOMINANT PLAN</span>{plan ? <dl><div><dt>Objective</dt><dd>{plan.objective}</dd></div><div><dt>Progress</dt><dd>{plan.steps_completed}/{plan.steps_total}</dd></div><div><dt>Completed</dt><dd>{plan.completed_operations.length ? plan.completed_operations.join(' → ') : 'None'}</dd></div><div><dt>Next</dt><dd>{plan.next_operation || 'None pending'}</dd></div><div><dt>Confidence</dt><dd>{plan.reasoning_confidence.toFixed(3)}</dd></div><div><dt>Uncertainty</dt><dd>{plan.uncertainties.length ? plan.uncertainties.join('; ') : 'None recorded'}</dd></div>{plan.last_outcome && <div><dt>Last outcome</dt><dd>{plan.last_outcome.operation}: {plan.last_outcome.success ? 'success' : 'failure'} · {plan.last_outcome.result}</dd></div>}{plan.decision_summary && <div><dt>Decision{plan.decision_revision ? ` r${plan.decision_revision}` : ''}</dt><dd>{plan.decision_summary}</dd></div>}</dl> : <p>No active operational plan.</p>}</div>
  </aside>;
}
const tabs = ['Overview', 'Activity', 'Diagnostics'] as const;

function IconButton({ label, children, active = false, className = '', ...props }: { label: string; children: ReactNode; active?: boolean; className?: string } & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  return <button {...props} className={`icon-button ${active ? 'active' : ''} ${className}`} aria-label={label} title={label}>{children}</button>;
}

function FluxDialogueWindow() {
  const dialogueId = new URLSearchParams(window.location.search).get('dialogue_id') || '';
  const [dialogue, setDialogue] = useState<{ child_name: string; topic: string; active: boolean; messages: DialogueMessage[] } | null>(null);
  const [error, setError] = useState('');
  const messagesRef = useRef<HTMLElement>(null);
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const response = await fetch(`/api/interface/network/dialogue/${encodeURIComponent(dialogueId)}`);
        if (response.status === 404 || response.status === 410) {
          if (!stopped) setError('This dialogue is no longer available. Start a new Flux dialogue from the main window.');
          return;
        }
        await checked(response);
        const data = await response.json();
        if (!stopped) { setDialogue(data); setError(''); }
        if (!data.active) return;
      } catch (e) { if (!stopped) setError(e instanceof Error ? e.message : 'Dialogue unavailable.'); }
      if (!stopped) timer = setTimeout(refresh, 1200);
    }
    if (dialogueId) void refresh(); else setError('No Flux dialogue selected.');
    return () => { stopped = true; clearTimeout(timer); };
  }, [dialogueId]);
  useEffect(() => { document.title = dialogue ? `Flux dialogue · ${dialogue.child_name}` : 'Flux dialogue'; }, [dialogue]);
  useEffect(() => { messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: 'smooth' }); }, [dialogue?.messages.length]);
  return <main className="dialogue-window"><header className="dialogue-window-heading"><div><span className="eyebrow">LUMINA NETWORK / FLUX</span><h1>Live dialogue{dialogue ? ` · ${dialogue.child_name}` : ''}</h1></div><span className={`network-indicator ${dialogue?.active ? 'online' : ''}`}><i/>{dialogue?.active ? 'In progress' : dialogue ? 'Complete' : 'Connecting'}</span></header><div className="dialogue-topic">{dialogue?.topic || 'Opening Flux conversation...'}</div><section ref={messagesRef} className="dialogue-messages" aria-live="polite">{error ? <p className="dialogue-error">{error}</p> : dialogue?.messages.length ? dialogue.messages.map((message, index) => <article className={`dialogue-message ${message.direction}`} key={`${message.timestamp}-${index}`}><div className="dialogue-author">{message.sender_name}<time>{new Date(message.timestamp * 1000).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</time></div><p>{message.text}</p></article>) : <p className="dialogue-empty">Waiting for the first exchange...</p>}</section></main>;
}

type CognitiveHealthData = Record<string, unknown>;

function displayValue(value: unknown): string {
  if (value === null || value === undefined) return 'Unavailable';
  if (typeof value === 'boolean') return value ? 'Yes' : 'No';
  if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(Math.abs(value) < 10 && value % 1 !== 0 ? 2 : 0) : 'Unavailable';
  if (typeof value === 'string') return value || 'Empty';
  if (Array.isArray(value)) return `${value.length} items`;
  return `${Object.keys(value as object).length} fields`;
}

const healthEmptyStates: Record<string, string> = {
  top: 'No identity themes synthesized yet',
  themes: 'No identity themes synthesized yet',
  rising: 'No dimensions are currently rising',
  falling: 'No dimensions are currently receding',
  recent_predictions: 'No qualifying predictions recorded',
  recent_drift: 'No recent identity drift detected',
  plans: 'No active long-horizon plans',
  do_not_repeat: 'No parameters suppressed by meta-learning',
  top_contexts: 'No learned contexts yet; feedback is needed to activate retrieval',
  active_hypotheses: 'No active workspace hypotheses',
  hypotheses: 'No active hypotheses recorded',
  recent: 'No recent events recorded',
  revisions: 'No self-modification decisions recorded',
  action_types: 'No causal action signatures recorded',
};

function emptyHealthState(field: string): string {
  return healthEmptyStates[field] || 'No records in this category';
}

function HealthValue({ value, field = '' }: { value: unknown; field?: string }) {
  if (value && typeof value === 'object') {
    if (Array.isArray(value)) return value.length ? <details className="health-detail" open><summary>{value.length} records</summary><div className="health-list">{value.slice(0, 12).map((item, index) => <div key={index}><HealthValue value={item}/></div>)}{value.length > 12 && <em>Showing 12 of {value.length}; full data is below.</em>}</div></details> : <em className="health-empty">{emptyHealthState(field)}</em>;
    const entries = Object.entries(value as Record<string, unknown>);
    return <details className="health-detail" open><summary>{entries.length} fields</summary><div className="health-object">{entries.slice(0, 18).map(([key, item]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><HealthValue value={item} field={key}/></div>)}{entries.length > 18 && <em>Showing 18 of {entries.length}; full data is below.</em>}</div></details>;
  }
  return <span className={`health-value ${typeof value === 'number' ? 'is-number' : typeof value === 'boolean' ? 'is-boolean' : ''}`}>{formatHealthTimestamp(field, value) ?? displayValue(value)}</span>;
}


function CognitiveHealth({ onData }: { onData?: (data: CognitiveHealthData) => void }) {
  const [data, setData] = useState<CognitiveHealthData | null>(null);
  const [error, setError] = useState('');
  const panels: { title: string; icon: string; keys: string[] }[] = [
    { title: 'Predictive Model Readiness', icon: '🧠', keys: ['pcm', 'wsdm', 'wsdm_network_records'] },
    { title: 'Identity Constraint Health', icon: '🛡', keys: ['ice', 'ice_by_value'] },
    { title: 'Cross-Layer Feedback', icon: '⚡', keys: ['ec_multiplier', 'clf', 'gate', 'arbitration'] },
    { title: 'Aspiration Pipeline', icon: '✦', keys: ['aspirations', 'tp', 'next_synthesis_cycle', 'next_generative_cycle', 'synthesis_locked', 'generative_locked'] },
    { title: 'Motivational State', icon: '⏱', keys: ['mf'] },
    { title: 'Resource Economy', icon: '⚡', keys: ['ec'] },
    { title: 'Lumina-Flux Network', icon: '🌐', keys: ['flux_relations', 'flux_mind'] },
    { title: 'Causal and Long-Horizon Planning', icon: '🔗', keys: ['causal_mechanism', 'long_horizon'] },
    { title: 'Narrative and Self-Trajectory', icon: '📖', keys: ['narrative_compression', 'temporal_self'] },
    { title: 'Meta-Learning and Structural Coupling', icon: '↪', keys: ['cognitive_audit', 'meta_learning', 'structural_coupling'] },
    { title: 'Cognitive Attention Allocation', icon: '🎯', keys: ['attention'] },
    { title: 'Contextual Attention Learning', icon: '⏱', keys: ['top_contexts', 'total_context_observations'] },
    { title: 'Calibration and Uncertainty', icon: '◢', keys: ['calibration'] },
    { title: 'Counterfactual Simulator', icon: '↪', keys: ['counterfactual'] },
    { title: 'Introspective Observer', icon: '🔭', keys: ['introspective'] },
    { title: 'Global Workspace', icon: '🌐', keys: ['global_workspace'] },
    { title: 'Emotion, Mood and Threat', icon: '🌡', keys: ['emotion_mood', 'threat_level'] },
    { title: 'Epistemic Self-Model', icon: '💡', keys: ['internal_cognitive_state'] },
    { title: 'Recursive Crossing (Self-Efficacy)', icon: '↻', keys: ['epistemic_efficacy'] },
    { title: 'Open Symbol Repertoire', icon: '🔣', keys: ['symbol_system'] },
    { title: 'Architecture as Object of Reflection', icon: '🧬', keys: ['self_description', 'revision_gateway'] },
    { title: 'Adaptive Reflection and Learned Weights', icon: '⚙', keys: ['reflection_controller', 'arbitration_learning'] },
    { title: 'Cognitive Self-Awareness Index', icon: '🎯', keys: ['self_awareness_index', 'loop_observability'] },
  ];
  useEffect(() => {
    let stopped = false;
    let timer: ReturnType<typeof setTimeout>;
    async function refresh() {
      try {
        const response = await checked(await fetch('/api/interface/cognitive-health'));
        const next = await response.json();
        if (!stopped) { setData(next); onData?.(next); setError(''); }
      } catch (e) { if (!stopped) setError(e instanceof Error ? e.message : 'Cognitive dashboard unavailable.'); }
      if (!stopped) timer = setTimeout(refresh, document.hidden ? 15000 : 5000);
    }
    void refresh();
    return () => { stopped = true; clearTimeout(timer); };
  }, []);
  const known = new Set([...panels.flatMap(panel => panel.keys), 'slow_cycle', 'fetched_at']);
  const metrics = data ? Object.entries(data).filter(([key, value]) => !known.has(key) && (typeof value !== 'object' || value === null)).slice(0, 8) : [];
  return <section className="cognitive-health-content">
    <header className="health-heading"><div><span className="eyebrow">LUMINA / COGNITIVE HEALTH</span><h2>Engine status, beneath the conversation</h2></div><div className="health-head-meta"><span className="health-cycle">Cycle #{data?.slow_cycle !== undefined ? displayValue(data.slow_cycle) : '—'}</span><span className="health-live"><i/>{error ? 'Refresh failed' : data ? 'Live telemetry' : 'Connecting'}</span><span>{data && formatHealthTimestamp('fetched_at', data.fetched_at)}</span></div></header>
    {error && <p className="health-error" role="status">{error}</p>}
    {metrics.length > 0 && <div className="health-metrics">{metrics.map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><strong>{displayValue(value)}</strong></div>)}</div>}
    <div className="health-columns">{[0, 1].map(column => <div className="health-column" key={column}>{panels.filter((_, index) => index % 2 === column).map((panel, index) => <article className={`health-panel tone-${(index + column) % 6}`} key={panel.title}><h3><span className="health-icon">{panel.icon}</span>{panel.title}<span className="health-index">{String(index * 2 + column + 1).padStart(2, '0')}</span></h3>{panel.keys.map(key => data && key in data ? <div className="health-row" key={key}><span>{key.replaceAll('_', ' ')}</span><HealthValue value={data[key]} field={key}/></div> : null)}{data && !panel.keys.some(key => key in data) && <p className="muted">No data recorded yet.</p>}{!data && <p className="muted">Waiting for engine telemetry.</p>}</article>)}</div>)}</div>
    <details className="health-raw"><summary>Raw engine data</summary><pre>{data ? JSON.stringify(data, null, 2) : 'Waiting for engine telemetry.'}</pre></details>
  </section>;
}

function CognitiveHealthWindow() { return <main className="health-window"><CognitiveHealth /></main>; }

type SettingsValue = string | number | boolean | null;
type SettingsTab = 'llm' | 'memory' | 'voice' | 'vision' | 'connector' | 'status' | 'lumina' | 'tools' | 'analytics' | 'development';
const settingTabs: { id: SettingsTab; label: string; icon: ReactNode }[] = [
  { id: 'llm', label: 'LLM', icon: <Sparkles size={16}/> }, { id: 'memory', label: 'Memory', icon: <Database size={16}/> },
  { id: 'voice', label: 'Voice', icon: <AudioLines size={16}/> }, { id: 'vision', label: 'Vision', icon: <Eye size={16}/> }, { id: 'connector', label: 'Connector', icon: <PlugZap size={16}/> },
  { id: 'status', label: 'Status', icon: <Activity size={16}/> }, { id: 'lumina', label: 'Lumina', icon: <HeartIcon/> },
  { id: 'tools', label: 'Tools', icon: <Settings size={16}/> }, { id: 'analytics', label: 'Analytics', icon: <History size={16}/> },
  { id: 'development', label: 'Development', icon: <Lightbulb size={16}/> },
];
function HeartIcon() { return <span aria-hidden="true">♥</span>; }
const settingsFields: Record<Exclude<SettingsTab, 'status' | 'lumina' | 'tools' | 'analytics' | 'development' | 'connector'>, { key: string; label: string; type?: 'select' | 'number' | 'textarea' | 'checkbox'; options?: string[] }[]> = {
  llm: [{ key: 'PERSONA_NAME', label: 'Persona name' }, { key: 'LLM_PROVIDER', label: 'Provider', type: 'select', options: ['ollama', 'lmstudio', 'openai'] }, { key: 'LLM_MODEL', label: 'Model' }, { key: 'TEXT_MODEL', label: 'Fast text model' }, { key: 'QUALITY_EMBED_MODEL', label: 'Quality embedding model' }, { key: 'EMBED_API_BASE_URL', label: 'Embedding API base URL' }, { key: 'AFFECT_EMBEDDING_MODEL', label: 'Multilingual affect embedding model' }, { key: 'LLM_BASE_URL', label: 'Base URL' }, { key: 'OPENAI_API_KEY', label: 'OpenAI API key' }, { key: 'ANTHROPIC_API_KEY', label: 'Anthropic API key' }, { key: 'RESEARCH_SEARCH_BACKEND', label: 'Research backend', type: 'select', options: ['auto', 'claude', 'brave', 'rss', 'stealth'] }, { key: 'BRAVE_SEARCH_KEY', label: 'Brave Search key' }, { key: 'SERPAPI_KEY', label: 'SerpAPI key' }, { key: 'RESPONSE_VERBOSITY', label: 'Response detail', type: 'select', options: ['concise', 'verbose'] }, { key: 'RESPONSE_TOKENS_CONCISE', label: 'Concise token budget', type: 'number' }, { key: 'RESPONSE_TOKENS_VERBOSE', label: 'Extended token budget', type: 'number' }, { key: 'CUSTOM_SYSTEM_PROMPT', label: 'Custom system prompt', type: 'textarea' }],
  memory: [{ key: 'MEMORY_BACKEND', label: 'Backend', type: 'select', options: ['faiss', 'dict', 'cognee', 'simple'] }, { key: 'MEMORY_DB_PATH', label: 'SQLite database path' }, { key: 'MEMORY_FAISS_PATH', label: 'FAISS index path' }, { key: 'MEMORY_WORLD_PATH', label: 'World model path' }, { key: 'MEMORY_PERSONA_PATH', label: 'Persona data folder' }, { key: 'MEMORY_COGNEE_PATH', label: 'Cognee data folder' }, { key: 'MEMORY_COGNEE_EMBED_MODEL', label: 'Cognee embedding model' }, { key: 'VOICEMEM_ENABLED', label: 'Enable optional VoiceMem', type: 'checkbox' }, { key: 'VOICEMEM_DATA_PATH', label: 'VoiceMem data folder' }, { key: 'VOICEMEM_TOP_K', label: 'VoiceMem retrieval count', type: 'number' }],
  voice: [{ key: 'TTS_PROVIDER', label: 'TTS provider', type: 'select', options: ['pyttsx3', 'edge-tts', 'elevenlabs', 'coqui', 'kokoro'] }, { key: 'STT_PROVIDER', label: 'STT provider', type: 'select', options: ['whisper', 'faster_whisper', 'openai'] }, { key: 'WHISPER_MODEL', label: 'Whisper model', type: 'select', options: ['tiny', 'base', 'small', 'medium', 'large'] }, { key: 'VOICE_LANGUAGE', label: 'TTS language' }, { key: 'RESPONSE_LANGUAGE', label: 'Response language' }, { key: 'ELEVENLABS_VOICE_ID', label: 'ElevenLabs voice ID' }, { key: 'COQUI_VOICE_REFERENCE', label: 'Coqui voice sample path' }, { key: 'TTS_STREAMING', label: 'Sentence streaming', type: 'checkbox' }, { key: 'TTS_BARGE_IN', label: 'Barge-in interruption', type: 'checkbox' }, { key: 'BARGE_IN_SENSITIVITY', label: 'Barge-in sensitivity', type: 'number' }, { key: 'VAD_AGGRESSIVENESS', label: 'VAD aggressiveness', type: 'number' }, { key: 'VAD_ONSET_CHUNKS', label: 'Onset confirmation chunks', type: 'number' }, { key: 'VAD_SILENCE_DURATION', label: 'Silence before end (seconds)', type: 'number' }, { key: 'VAD_MIN_SPEECH_DURATION', label: 'Minimum speech (seconds)', type: 'number' }, { key: 'VAD_ENERGY_GATE_FACTOR', label: 'Energy gate factor', type: 'number' }, { key: 'INTER_SENTENCE_PAUSE_MS', label: 'Inter-sentence pause (ms)', type: 'number' }, { key: 'TTS_POST_ROLL_MS', label: 'Echo post-roll (ms)', type: 'number' }],
  vision: [{ key: 'CAMERA_AUTOSTART', label: 'Camera on startup', type: 'checkbox' }, { key: 'CAMERA_ID', label: 'Camera ID', type: 'number' }, { key: 'CAMERA_FPS', label: 'Frame rate', type: 'number' }, { key: 'CAMERA_RESOLUTION', label: 'Resolution', type: 'select', options: ['640x480', '1280x720'] }, { key: 'VISION_MODE', label: 'Vision mode', type: 'select', options: ['keyword', 'always', 'context'] }, { key: 'LAVA_MODEL', label: 'Vision model' }, { key: 'VISION_LLM_MODE', label: 'Vision routing', type: 'select', options: ['separate', 'direct'] }, { key: 'AMBIENT_VISION_INTERVAL', label: 'Ambient interval (seconds)', type: 'number' }],
};
const voiceLanguages = [['en', 'EN - English'], ['fr', 'FR - Francais'], ['uk', 'UK - Ukrainian'], ['de', 'DE - German'], ['es', 'ES - Spanish'], ['it', 'IT - Italian'], ['pt', 'PT - Portuguese'], ['ru', 'RU - Russian'], ['nl', 'NL - Dutch'], ['pl', 'PL - Polish'], ['ja', 'JA - Japanese'], ['ko', 'KO - Korean'], ['zh-cn', 'ZH - Chinese']];
settingsFields.llm.unshift({ key: 'NICEGUI_HOST', label: 'Interface access', type: 'select', options: ['127.0.0.1', '0.0.0.0'] });
const settingOptionLabel = (key: string, option: string) => key === 'RESPONSE_VERBOSITY'
  ? (option === 'verbose' ? 'Extended' : 'Concise')
  : option;

function SettingsSubpage({ tab, children, compact = false }: { tab: SettingsTab; children: ReactNode; compact?: boolean }) {
  const current = settingTabs.find(item => item.id === tab);
  return <div className={`settings-subpage ${compact ? 'settings-subpage-compact' : ''}`}><div className="settings-subpage-heading"><div><span className="eyebrow">LUMINA / LEGACY MIGRATION</span><h3>{current?.icon} {current?.label}</h3></div><span className="live-badge"><i/>Live engine data</span></div><nav className="settings-subnav" aria-label="Settings sections">{settingTabs.map(item => <button className={item.id === tab ? 'active' : ''} key={item.id} onClick={() => { window.history.replaceState({}, '', `/next/?view=settings&tab=${item.id}`); window.location.reload(); }}>{item.icon}<span>{item.label}</span></button>)}</nav>{children}</div>;
}

function SettingsMemoryComplete() {
  const [values, setValues] = useState<Record<string, any>>({});
  const [message, setMessage] = useState('');
  const [voiceMemStatus, setVoiceMemStatus] = useState<any>(null);
  const set = (key: string, value: any) => setValues(current => ({ ...current, [key]: value }));
  useEffect(() => {
    void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => setValues(r.values || {})).catch(e => setMessage(e instanceof Error ? e.message : 'Memory settings unavailable.'));
    void fetch('/api/interface/voicemem/status').then(checked).then(r => r.json()).then(setVoiceMemStatus).catch(() => {});
  }, []);
  const save = async () => {
    try {
      const response = await checked(await fetch('/api/interface/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) }));
      const result = await response.json();
      setValues(result.values || values);
      setMessage('Saved. Restart recommended for memory backend changes.');
    } catch (e) {
      setMessage(e instanceof Error ? e.message : 'Memory settings could not be saved.');
    }
  };
  return <SettingsSubpage tab="memory"><div className="settings-section-title">Memory substrate</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>Backend</span><select value={values.MEMORY_BACKEND || 'faiss'} onChange={e => set('MEMORY_BACKEND', e.target.value)}><option>faiss</option><option>dict</option><option>cognee</option><option>simple</option></select></label><label className="settings-field"><span>SQLite database path</span><input value={values.MEMORY_DB_PATH || ''} onChange={e => set('MEMORY_DB_PATH', e.target.value)}/></label><label className="settings-field"><span>FAISS index path</span><input value={values.MEMORY_FAISS_PATH || ''} onChange={e => set('MEMORY_FAISS_PATH', e.target.value)}/></label><label className="settings-field"><span>World model path</span><input value={values.MEMORY_WORLD_PATH || ''} onChange={e => set('MEMORY_WORLD_PATH', e.target.value)}/></label><label className="settings-field"><span>Persona data folder</span><input value={values.MEMORY_PERSONA_PATH || ''} onChange={e => set('MEMORY_PERSONA_PATH', e.target.value)}/></label><label className="settings-field"><span>Cognee data folder</span><input value={values.MEMORY_COGNEE_PATH || ''} onChange={e => set('MEMORY_COGNEE_PATH', e.target.value)}/></label><label className="settings-field"><span>Cognee embedding model</span><input value={values.MEMORY_COGNEE_EMBED_MODEL || ''} onChange={e => set('MEMORY_COGNEE_EMBED_MODEL', e.target.value)}/></label></div><div className="settings-section-title">Optional VoiceMem</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>Enable VoiceMem</span><input type="checkbox" checked={Boolean(values.VOICEMEM_ENABLED)} disabled={!voiceMemStatus?.available && !values.VOICEMEM_ENABLED} onChange={e => set('VOICEMEM_ENABLED', e.target.checked)}/></label><label className="settings-field"><span>Data folder</span><input value={values.VOICEMEM_DATA_PATH || ''} onChange={e => set('VOICEMEM_DATA_PATH', e.target.value)}/></label><label className="settings-field"><span>Retrieval count</span><input type="number" min="1" max="20" value={values.VOICEMEM_TOP_K ?? 5} onChange={e => set('VOICEMEM_TOP_K', Number(e.target.value))}/></label></div><p className="muted">{voiceMemStatus?.available ? 'VoiceMem package detected.' : 'VoiceMem package not installed. Install it before enabling this setting.'} {voiceMemStatus?.error && `(${voiceMemStatus.error})`}</p><div className="lumina-actions"><button onClick={() => void save()}><Check size={15}/>Save memory settings</button><span>{message}</span></div></SettingsSubpage>;
}

function SettingsUniversalConnector() {
  const [values, setValues] = useState<Record<string, any>>({});
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  const [discovery, setDiscovery] = useState<HomeAssistantDiscovery | null>(null);
  useEffect(() => {
    void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => setValues(r.values || {})).catch(e => setError(e instanceof Error ? e.message : 'Connector settings unavailable.'));
  }, []);
  const set = (key: string, value: any) => setValues(current => ({ ...current, [key]: value }));
  const save = async () => {
    setMessage('Saving...'); setError('');
    try {
      const response = await checked(await fetch('/api/interface/settings', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ values }),
      }));
      const result = await response.json();
      setValues(result.values || values);
      setMessage('Saved to config.json. Restart recommended for connector changes.');
    } catch (e) { setMessage(''); setError(e instanceof Error ? e.message : 'Connector settings could not be saved.'); }
  };
  const discover = async () => { setDiscovery(null); setMessage('Checking Home Assistant...'); setError(''); try { const response = await checked(await fetch('/api/interface/connector/home-assistant/discover', { method: 'POST' })); const result: HomeAssistantDiscovery = await response.json(); setDiscovery(result); setMessage(result.ok ? `Connected. ${result.count ?? 0} permitted entities found.` : ''); if (!result.ok) setError(result.message || 'Home Assistant discovery failed.'); } catch (e) { setMessage(''); setError(e instanceof Error ? e.message : 'Home Assistant discovery failed.'); } };
  return <SettingsSubpage tab="connector"><div className="settings-section-title">Universal Connector</div><p className="muted">A local gateway for sensors, services, and future actuators. It keeps external tools separate from the cognitive core and does not connect until enabled.</p><div className="settings-form settings-form-compact"><label className="settings-field"><span>Enable Universal Connector</span><input type="checkbox" checked={Boolean(values.UNIVERSAL_CONNECTOR_ENABLED)} onChange={e => set('UNIVERSAL_CONNECTOR_ENABLED', e.target.checked)}/></label><label className="settings-field"><span>Enable Home Assistant bridge</span><input type="checkbox" checked={Boolean(values.HOME_ASSISTANT_ENABLED)} onChange={e => set('HOME_ASSISTANT_ENABLED', e.target.checked)}/></label><label className="settings-field"><span>Use Home Assistant for presence</span><input type="checkbox" checked={Boolean(values.HOME_ASSISTANT_PRESENCE_ENABLED)} onChange={e => set('HOME_ASSISTANT_PRESENCE_ENABLED', e.target.checked)}/></label><label className="settings-field"><span>Home Assistant URL</span><input value={values.HOME_ASSISTANT_URL || ''} onChange={e => set('HOME_ASSISTANT_URL', e.target.value)} placeholder="http://192.168.0.38:8123"/></label><label className="settings-field"><span>Long-lived access token</span><input type="password" value={values.HOME_ASSISTANT_TOKEN === '••••••••' ? '' : (values.HOME_ASSISTANT_TOKEN || '')} onChange={e => set('HOME_ASSISTANT_TOKEN', e.target.value)} placeholder="Stored securely in config.json"/></label><label className="settings-field"><span>Verify HTTPS certificate</span><input type="checkbox" checked={values.HOME_ASSISTANT_VERIFY_SSL !== false} onChange={e => set('HOME_ASSISTANT_VERIFY_SSL', e.target.checked)}/></label><label className="settings-field"><span>Polling interval (seconds)</span><input type="number" min="1" max="300" value={values.HOME_ASSISTANT_POLL_INTERVAL ?? 5} onChange={e => set('HOME_ASSISTANT_POLL_INTERVAL', Number(e.target.value))}/></label><label className="settings-field"><span>Entities presented to Lumina (comma-separated)</span><input value={values.HOME_ASSISTANT_SELECTED_ENTITIES || ''} onChange={e => set('HOME_ASSISTANT_SELECTED_ENTITIES', e.target.value)} placeholder="Leave empty for all permitted entities"/></label><label className="settings-field wide"><span>Allowed Home Assistant domains</span><input value={values.HOME_ASSISTANT_ALLOWED_DOMAINS || ''} onChange={e => set('HOME_ASSISTANT_ALLOWED_DOMAINS', e.target.value)} placeholder="person,device_tracker,sensor,binary_sensor,zone"/></label></div><div className="provider-notes"><div>HOME ASSISTANT</div><p>Use the NAS Home Assistant address and create a Long-Lived Access Token in the Home Assistant profile.</p><p>Discovery is read-only. Initially, Lumina will treat permitted entities as perceptions; actuator control remains separately permissioned and auditable.</p></div><div className="lumina-actions"><button onClick={() => void save()}><Check size={15}/>Save connector settings</button><button className="settings-action-button" onClick={() => void discover()}><PlugZap size={15}/>Test and discover</button><span>{message || error}</span></div>{discovery?.entities?.length ? <><div className="settings-section-title">Discovered entities</div><div className="settings-event-list">{discovery.entities.map(entity => <div key={entity.entity_id}><span>{entity.friendly_name || entity.entity_id}<small>{entity.entity_id}{entity.device_class ? ` · ${entity.device_class}` : ''}</small></span><b>{entity.signal === 'presence_detected' ? 'Presence detected' : entity.signal === 'no_presence' ? 'No presence' : `${entity.state}${entity.unit ? ` ${entity.unit}` : ''}`}</b></div>)}</div></> : discovery && <p className="muted">{discovery.message || 'No permitted entities found.'}</p>}</SettingsSubpage>;
}

function SettingsStatus() {
  const [data, setData] = useState<Status | null>(null);
  useEffect(() => { let live = true; const read = () => void fetch('/api/interface/status').then(checked).then(r => r.json()).then(v => live && setData(v)).catch(() => {}); read(); const timer = setInterval(read, 4000); return () => { live = false; clearInterval(timer); }; }, []);
  return <SettingsSubpage tab="status"><div className="settings-stat-grid">{[['Runtime', data?.ready ? 'Ready' : 'Starting'], ['Cycle', data?.cycle ?? 'Unavailable'], ['Uptime', data ? `${Math.floor(data.uptime / 60)} min` : 'Unavailable'], ['Language', data?.response_language || 'auto'], ['STT', data?.providers.stt || 'Unavailable'], ['TTS', data?.providers.tts || 'Unavailable']].map(([label, value]) => <div className="settings-stat" key={label}><span>{label}</span><strong>{value}</strong></div>)}</div><div className="settings-section-title">Runtime modules</div><div className="settings-module-list">{Object.entries(data?.engines || {}).map(([name, ready]) => <div key={name}><span>{name}</span><b className={ready ? 'ok' : ''}>{ready ? 'Available' : 'Unavailable'}</b></div>)}</div><div className="settings-section-title">Recent events</div><div className="settings-event-list">{(data?.events || []).slice(-8).reverse().map(event => <div key={event.id}><span>{event.label}</span><time>{new Date(event.timestamp * 1000).toLocaleTimeString()}</time></div>)}</div></SettingsSubpage>;
}

function SettingsLumina() {
  const [data, setData] = useState<Record<string, any> | null>(null); const [message, setMessage] = useState('');
  const [location, setLocation] = useState<Record<string, any>>({});
  const read = () => void fetch('/api/interface/lumina').then(checked).then(r => r.json()).then(setData).catch(e => setMessage(e instanceof Error ? e.message : 'Unavailable'));
  useEffect(() => { read(); void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => setLocation(r.values || {})).catch(() => {}); const timer = setInterval(read, 5000); return () => clearInterval(timer); }, []);
  const action = async (name: string) => { setMessage('Running...'); try { await checked(await fetch('/api/interface/lumina/action', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ action: name }) })); setMessage('Cycle complete.'); read(); } catch (e) { setMessage(e instanceof Error ? e.message : 'Action failed.'); } };
  const saveLocation = async () => { try { const response = await checked(await fetch('/api/interface/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values: { LOCATION_NAME: location.LOCATION_NAME || '', LOCATION_ADDRESS: location.LOCATION_ADDRESS || '', LOCATION_LATITUDE: location.LOCATION_LATITUDE === '' ? null : location.LOCATION_LATITUDE, LOCATION_LONGITUDE: location.LOCATION_LONGITUDE === '' ? null : location.LOCATION_LONGITUDE } }) })); const result = await response.json(); setLocation(result.values || location); setMessage('Location saved.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Location could not be saved.'); } };
  const emotion = data?.emotional_state || {};
  return <SettingsSubpage tab="lumina"><div className="lumina-actions"><button onClick={() => void action('dream')}><Sparkles size={15}/>Dream cycle</button><button onClick={() => void action('learning')}><Activity size={15}/>Learning cycle</button><button onClick={() => void action('life_event')}><History size={15}/>Life event</button><span>{message}</span></div><div className="settings-stat-grid">{[['Life stage', data?.life_stage], ['Age', data?.age], ['Interactions', data?.interaction_count], ['Active user', data?.active_user || 'None'], ['Memory records', data?.memory_count], ['LLM', data?.llm_available ? 'Available' : 'Unavailable']].map(([label, value]) => <div className="settings-stat" key={label}><span>{label}</span><strong>{value ?? 'Unavailable'}</strong></div>)}</div><div className="settings-section-title">Current physical location</div><p className="muted">Used as a changeable world-model anchor. Coordinates are used for distance and travel reasoning when provided.</p><div className="settings-form settings-form-compact"><label className="settings-field"><span>Location name</span><input value={location.LOCATION_NAME || ''} onChange={e => setLocation(v => ({ ...v, LOCATION_NAME: e.target.value }))}/></label><label className="settings-field"><span>Address</span><input value={location.LOCATION_ADDRESS || ''} onChange={e => setLocation(v => ({ ...v, LOCATION_ADDRESS: e.target.value }))}/></label><label className="settings-field"><span>Latitude</span><input type="number" step="any" min="-90" max="90" value={location.LOCATION_LATITUDE ?? ''} onChange={e => setLocation(v => ({ ...v, LOCATION_LATITUDE: e.target.value }))}/></label><label className="settings-field"><span>Longitude</span><input type="number" step="any" min="-180" max="180" value={location.LOCATION_LONGITUDE ?? ''} onChange={e => setLocation(v => ({ ...v, LOCATION_LONGITUDE: e.target.value }))}/></label></div><div className="lumina-actions"><button onClick={() => void saveLocation()}><Check size={15}/>Save location</button></div><div className="settings-section-title">Emotional state</div><div className="settings-object-grid">{Object.entries(emotion).slice(0, 10).map(([key, value]) => <div key={key}><span>{key.replaceAll('_', ' ')}</span><b>{typeof value === 'number' ? value.toFixed(2) : String(value)}</b></div>)}</div><div className="settings-section-title">Identity and self-concept</div><pre className="settings-json">{JSON.stringify({ identity: data?.identity, self_concept: data?.self_concept, evolution: data?.evolution }, null, 2)}</pre></SettingsSubpage>;
}

function SettingsTools() {
  const [data, setData] = useState<any>(null); useEffect(() => { void fetch('/api/interface/settings/tools').then(checked).then(r => r.json()).then(setData); }, []);
  return <SettingsSubpage tab="tools"><div className="settings-stat-grid">{Object.entries(data?.engines || {}).map(([label, value]) => <div className="settings-stat" key={label}><span>{label}</span><strong className={value ? 'value-ok' : ''}>{value ? 'Ready' : 'Offline'}</strong></div>)}</div><div className="settings-section-title">Observatory</div><div className="settings-object-grid"><div><span>Configured</span><b>{data?.observatory?.enabled ? 'Enabled' : 'Disabled'}</b></div><div><span>Collecting</span><b>{data?.observatory?.active ? 'Active' : 'Inactive'}</b></div><div><span>Security</span><b className="value-ok">Local only</b></div></div><div className="settings-section-title">Recent tool events</div><div className="settings-event-list">{(data?.events || []).slice().reverse().map((event: any) => <div key={event.id}><span>{event.label}</span><time>{new Date(event.timestamp * 1000).toLocaleTimeString()}</time></div>)}</div></SettingsSubpage>;
}

function SettingsAnalytics() {
  const [data, setData] = useState<any>(null); useEffect(() => { void fetch('/api/interface/settings/analytics').then(checked).then(r => r.json()).then(setData); }, []);
  const history = data?.history || []; const latest = history[history.length - 1] || {};
  return <SettingsSubpage tab="analytics"><div className="settings-stat-grid">{[['Observatory ticks', history.length], ['Log lines', data?.log?.lines ?? 0], ['Errors', data?.log?.errors ?? 0], ['Warnings', data?.log?.warnings ?? 0], ['Latest cycle', latest.cycle ?? 'Unavailable']].map(([label, value]) => <div className="settings-stat" key={label}><span>{label}</span><strong className={label === 'Errors' && Number(value) > 0 ? 'value-warn' : ''}>{value}</strong></div>)}</div><div className="settings-section-title">Cognitive trend snapshot</div><div className="analytics-metrics">{['ccs', 'gei', 'idx', 'str_score'].map(key => { const values = history.map((item: any) => Number(item[key] || 0)); const current = values.at(-1) ?? 0; const change = current - (values.at(-2) ?? current); return <div key={key}><span>{key.toUpperCase()}</span><b>{current.toFixed(3)}</b><em className={change >= 0 ? 'trend-up' : 'trend-down'}>{change >= 0 ? '↑' : '↓'} {Math.abs(change).toFixed(3)}</em></div>; })}</div><div className="settings-section-title">Recent errors</div>{data?.log?.recent_errors?.length ? <pre className="settings-log">{data.log.recent_errors.join('\n')}</pre> : <p className="muted">No errors in the last 500 log lines.</p>}</SettingsSubpage>;
}

function SettingsToolsFull() {
  const [data, setData] = useState<any>(null); const [busy, setBusy] = useState(false);
  const read = () => void fetch('/api/interface/settings/tools').then(checked).then(r => r.json()).then(setData).catch(() => {});
  useEffect(() => { read(); const timer = setInterval(read, 5000); return () => clearInterval(timer); }, []);
  return <SettingsSubpage tab="tools"><div className="settings-section-title">Cognitive Observatory</div><div className="settings-stat-grid">{[['Enabled', data?.observatory?.enabled ? 'Yes' : 'No'], ['Collecting', data?.observatory?.active ? 'Active' : 'Inactive'], ['Baseline samples', data?.observatory?.baseline_samples], ['Emergence threshold', data?.observatory?.emergence_threshold]].map(([label, value]) => <div className="settings-stat" key={label}><span>{label}</span><strong>{value ?? 'Unavailable'}</strong></div>)}</div><div className="settings-section-title">Research and MCP</div><div className="settings-object-grid"><div><span>MCP status</span><b className={data?.research?.mcp_available ? 'value-ok' : ''}>{data?.research?.mcp_available ? 'Available' : 'Unavailable'}</b></div><div><span>Search backend</span><b>{data?.research?.backend || 'auto'}</b></div><div><span>Brave key</span><b>{data?.research?.brave_configured ? 'Configured' : 'Not configured'}</b></div></div><div className="settings-section-title">Security audit</div>{data?.security?.warnings?.length ? <div className="settings-warning-list">{data.security.warnings.map((item: string, index: number) => <div key={index}>Warning: {item}</div>)}</div> : <p className="settings-ok-line">No configuration warnings.</p>}<details className="settings-disclosure"><summary>Asimov safety layers</summary><p>Input filtering, output filtering, immutable identity beliefs, rate limiting, injection detection, and append-only audit logging.</p></details><div className="settings-section-title">Vision diagnostics</div><div className="settings-object-grid"><div><span>Camera</span><b className={data?.vision?.camera_active ? 'value-ok' : ''}>{data?.vision?.camera_active ? 'Active' : 'Off'}</b></div><div><span>Vision model</span><b>{data?.vision?.model || 'Unavailable'}</b></div><div><span>Routing</span><b>{data?.vision?.llm_mode || 'separate'}</b></div><div><span>Ambient interval</span><b>{data?.vision?.ambient_interval ?? 'Unavailable'}s</b></div></div><div className="settings-section-title">Tool actions</div><button className="settings-action-button" disabled={busy} onClick={() => { setBusy(true); read(); setTimeout(() => setBusy(false), 500); }}><RefreshCw size={15}/>Refresh diagnostics</button><div className="settings-section-title">Recent security events</div><pre className="settings-log">{(data?.security?.audit_events || []).map((item: any) => JSON.stringify(item)).join('\n') || 'No audit events recorded.'}</pre></SettingsSubpage>;
}

function SettingsAnalyticsFull() {
  const [data, setData] = useState<any>(null); const [analysis, setAnalysis] = useState(''); const [busy, setBusy] = useState(false);
  const read = () => void fetch('/api/interface/settings/analytics').then(checked).then(r => r.json()).then(setData).catch(() => {});
  useEffect(() => { read(); const timer = setInterval(read, 6000); return () => clearInterval(timer); }, []);
  const analyze = async () => { setBusy(true); try { const r = await checked(await fetch('/api/interface/settings/analytics/self-analysis', { method: 'POST' })); setAnalysis((await r.json()).analysis || ''); } catch (e) { setAnalysis(e instanceof Error ? e.message : 'Analysis unavailable.'); } finally { setBusy(false); } };
  return <SettingsSubpage tab="analytics"><div className="settings-stat-grid">{[['Ticks', data?.history?.length || 0], ['Info lines', data?.log?.infos || 0], ['Warnings', data?.log?.warnings || 0], ['Errors', data?.log?.errors || 0], ['Memory concepts', data?.memory?.concepts ?? 'Unavailable'], ['Memory relations', data?.memory?.relations ?? 'Unavailable']].map(([label, value]) => <div className="settings-stat" key={label}><span>{label}</span><strong className={label === 'Errors' && Number(value) > 0 ? 'value-warn' : ''}>{value}</strong></div>)}</div><div className="settings-section-title">Observatory metric trends</div><div className="analytics-metrics">{Object.entries(data?.trends || {}).map(([key, item]: [string, any]) => <div key={key}><span>{key.toUpperCase()}</span><b>{Number(item.current).toFixed(3)}</b><em className={item.delta >= 0 ? 'trend-up' : 'trend-down'}>{item.delta >= 0 ? '↑' : '↓'} {Math.abs(item.delta).toFixed(3)} · avg {Number(item.average).toFixed(3)}</em><small>range {Number(item.minimum).toFixed(3)}–{Number(item.maximum).toFixed(3)}</small></div>)}</div><div className="settings-section-title">Log pattern scanner</div><div className="settings-object-grid"><div><span>Window</span><b>Last 500 lines</b></div><div><span>Errors</span><b className={data?.log?.errors ? 'value-warn' : 'value-ok'}>{data?.log?.errors || 0}</b></div><div><span>Warnings</span><b>{data?.log?.warnings || 0}</b></div></div>{data?.log?.recent_errors?.length ? <pre className="settings-log">{data.log.recent_errors.join('\n')}</pre> : <p className="settings-ok-line">No errors in the current log window.</p>}<div className="settings-section-title">Lumina self-analysis</div><button className="settings-action-button" disabled={busy} onClick={() => void analyze()}><Sparkles size={15}/>{busy ? 'Analyzing...' : 'Analyze cognitive trends'}</button>{analysis && <div className="settings-analysis">{analysis}</div>}</SettingsSubpage>;
}

function SettingsVoiceFull() {
  const [values, setValues] = useState<Record<string, any>>({}); const [message, setMessage] = useState('');
  useEffect(() => { void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => setValues(r.values || {})); }, []);
  const set = (key: string, value: any) => setValues(current => ({ ...current, [key]: value }));
  const save = async () => { try { await checked(await fetch('/api/interface/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) })); setMessage('Saved. Restart recommended for provider changes.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Save failed.'); } };
  const test = async () => { try { const response = await checked(await fetch('/api/interface/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: `This is a ${values.PERSONA_NAME || 'Lumina'} voice test.` }) })); const blob = await response.blob(); const audio = new Audio(URL.createObjectURL(blob)); void audio.play(); setMessage('Voice test started.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Voice test failed.'); } };
  return <SettingsSubpage tab="voice"><div className="settings-section-title">TTS engine</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>Provider</span><select value={values.TTS_PROVIDER || ''} onChange={e => set('TTS_PROVIDER', e.target.value)}><option>pyttsx3</option><option>coqui</option><option>kokoro</option><option>edge-tts</option><option>elevenlabs</option></select></label><label className="settings-field"><span>Language</span><input value={values.VOICE_LANGUAGE || ''} onChange={e => set('VOICE_LANGUAGE', e.target.value)}/></label></div>{values.TTS_PROVIDER === 'coqui' && <div className="conditional-panel coqui-panel"><h4>Coqui voice cloning</h4><p>Use a WAV reference sample for the cloned voice. The legacy recording workflow writes to this path.</p><label className="settings-field"><span>Voice sample path</span><input value={values.COQUI_VOICE_REFERENCE || ''} onChange={e => set('COQUI_VOICE_REFERENCE', e.target.value)}/></label><div className="coqui-script"><b>Read-aloud sample</b><p>The quick brown fox jumps over the lazy dog near the riverbank. Every morning, I wake up and look out the window at the sky. Sometimes it is bright and clear; other times, clouds gather on the horizon. I enjoy a warm cup of coffee while thinking about the day ahead.</p></div><button className="settings-action-button" onClick={() => void test()}><Volume2 size={15}/>Test Coqui voice</button></div>}<div className="settings-section-title">Always-listen and echo control</div><div className="settings-form settings-form-compact">{[['TTS_STREAMING','Sentence streaming'],['TTS_BARGE_IN','Barge-in interruption']].map(([key,label]) => <label className="settings-field" key={key}><span>{label}</span><input type="checkbox" checked={Boolean(values[key])} onChange={e => set(key, e.target.checked)}/></label>)}{[['VAD_AGGRESSIVENESS','VAD aggressiveness'],['VAD_ONSET_CHUNKS','Onset chunks'],['VAD_SILENCE_DURATION','Silence duration'],['VAD_MIN_SPEECH_DURATION','Minimum speech'],['VAD_ENERGY_GATE_FACTOR','Energy gate'],['INTER_SENTENCE_PAUSE_MS','Sentence pause'],['TTS_POST_ROLL_MS','Echo post-roll']].map(([key,label]) => <label className="settings-field" key={key}><span>{label}</span><input type="number" value={values[key] ?? ''} onChange={e => set(key, Number(e.target.value))}/></label>)}</div><div className="lumina-actions"><button onClick={() => void save()}><Check size={15}/>Save voice settings</button><span>{message}</span></div></SettingsSubpage>;
}

function SettingsLlmFull() {
  const [values, setValues] = useState<Record<string, any>>({}); const [secrets, setSecrets] = useState<string[]>([]); const [message, setMessage] = useState('');
  useEffect(() => { void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => { setValues(r.values || {}); setSecrets(r.secret_fields || []); }); }, []);
  const set = (key: string, value: any) => setValues(current => ({ ...current, [key]: value }));
  const save = async () => { try { await checked(await fetch('/api/interface/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) })); setMessage('Saved. Restart required for model and provider changes.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Save failed.'); } };
  return <SettingsSubpage tab="llm"><div className="settings-form settings-form-compact">{settingsFields.llm.map(field => <label className={`settings-field ${field.type === 'textarea' ? 'wide' : ''}`} key={field.key}><span>{field.label}</span>{field.type === 'textarea' ? <textarea value={values[field.key] || ''} onChange={e => set(field.key, e.target.value)}/> : field.type === 'select' ? <select value={values[field.key] || ''} onChange={e => set(field.key, e.target.value)}>{field.options?.map(option => <option value={option} key={option}>{settingOptionLabel(field.key, option)}</option>)}</select> : <input type={secrets.includes(field.key) ? 'password' : field.type === 'number' ? 'number' : 'text'} value={values[field.key] ?? ''} onChange={e => set(field.key, field.type === 'number' ? Number(e.target.value) : e.target.value)}/>}</label>)}</div><div className="provider-notes"><div>PROVIDER NOTES</div><p><b>ollama</b> → <code>http://localhost:11434</code> · run <code>ollama serve</code> first</p><p><b>lmstudio</b> → <code>http://localhost:1234/v1</code> · start LM Studio server</p><p><b>openai</b> → requires API key above</p><div>AFFECT EMBEDDING MODEL</div><p>This is a local multilingual sentence model for background affect estimation, not a generative/chat LLM. The selected model must be cached locally; changing it requires a restart.</p></div><div className="lumina-actions"><button onClick={() => void save()}><Check size={15}/>Save LLM settings</button><span>{message}</span></div></SettingsSubpage>;
}

function VisionFeed({ data }: { data: any }) {
  const [feed, setFeed] = useState<any>(null);
  useEffect(() => { let live = true; const read = () => void fetch('/api/interface/camera/frame').then(r => r.json()).then(v => live && setFeed(v)).catch(() => {}); read(); const timer = setInterval(read, 450); return () => { live = false; clearInterval(timer); }; }, []);
  return <div className="vision-feed-panel"><div className="settings-section-title">Live camera feed</div><div className="vision-feed-frame">{feed?.frame ? <img src={`data:image/jpeg;base64,${feed.frame}`} alt="Live camera feed"/> : <div className="vision-feed-empty"><Eye size={24}/><span>Start the camera to see the live feed.</span></div>}{(feed?.faces || []).map((face: any, index: number) => { const loc = face.location || {}; return <div className="vision-feed-face" key={index} style={{ left: `${(loc.left || 0) / 6.4}%`, top: `${(loc.top || 0) / 4.8}%`, width: `${((loc.right || 0) - (loc.left || 0)) / 6.4}%`, height: `${((loc.bottom || 0) - (loc.top || 0)) / 4.8}%` }}><b>{face.name} · {Math.round((face.confidence || 0) * 100)}%</b></div>; })}</div><div className="vision-feed-meta"><span>{feed?.active ? 'Camera active' : 'Camera off'}</span><span>{feed?.faces?.length || 0} detected faces</span></div></div>;
}

function SettingsVisionComplete() {
  const [values, setValues] = useState<Record<string, any>>({}); const [data, setData] = useState<any>(null); const [name, setName] = useState(''); const [message, setMessage] = useState('');
  const read = () => void fetch('/api/interface/settings/tools').then(checked).then(r => r.json()).then(r => setData(r.vision));
  useEffect(() => { void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => setValues(r.values || {})); read(); }, []);
  const set = (key: string, value: any) => setValues(v => ({ ...v, [key]: value }));
  const apply = async (face?: boolean) => { try { const r = await checked(await fetch('/api/interface/vision/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ camera_id: Number(values.CAMERA_ID ?? 0), fps: Number(values.CAMERA_FPS ?? 5), resolution: values.CAMERA_RESOLUTION || '640x480', face_detection: face }) })); setData(await r.json()); setMessage('Vision settings applied.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Camera settings failed.'); } };
  const capture = async () => { try { const r = await checked(await fetch('/api/interface/vision/capture-face', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) })); setMessage((await r.json()).message); setName(''); read(); } catch (e) { setMessage(e instanceof Error ? e.message : 'Face capture failed.'); } };
  return <SettingsSubpage tab="vision"><VisionFeed data={data}/><div className="settings-section-title">Camera capture</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>Camera selection</span><select value={values.CAMERA_ID ?? 0} onChange={e => set('CAMERA_ID', Number(e.target.value))}><option value="0">Camera 0</option><option value="1">Camera 1</option><option value="2">Camera 2</option></select></label><label className="settings-field"><span>Frame rate</span><select value={values.CAMERA_FPS ?? 5} onChange={e => set('CAMERA_FPS', Number(e.target.value))}><option value="5">5 fps</option><option value="10">10 fps</option><option value="15">15 fps</option></select></label><label className="settings-field"><span>Resolution</span><select value={values.CAMERA_RESOLUTION || '640x480'} onChange={e => set('CAMERA_RESOLUTION', e.target.value)}><option>640x480</option><option>1280x720</option></select></label></div><div className="lumina-actions"><button onClick={() => void apply()}><Check size={15}/>Apply capture settings</button><button onClick={() => void fetch('/api/interface/camera/toggle', { method: 'POST' }).then(read)}><Eye size={15}/>Test camera</button></div><div className="settings-section-title">Face recognition capture</div><div className="conditional-panel"><label className="settings-field"><span>Enable face recognition</span><input type="checkbox" checked={Boolean(data?.face_detection)} onChange={e => void apply(e.target.checked)}/></label><div className="vision-capture-row"><input placeholder="Name for the current face" value={name} onChange={e => setName(e.target.value)}/><button disabled={!name.trim() || !data?.camera_active} onClick={() => void capture()}><Eye size={15}/>Capture face</button></div><p className="muted">Known faces: {data?.known_faces ?? 0} · Recognition engine: {data?.initialized ? 'Initialized' : 'Unavailable'}</p></div><div className="settings-section-title">Vision memory and model</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>LAVA model</span><input value={values.LAVA_MODEL || ''} onChange={e => set('LAVA_MODEL', e.target.value)}/></label><label className="settings-field"><span>LLM routing</span><select value={values.VISION_LLM_MODE || 'separate'} onChange={e => set('VISION_LLM_MODE', e.target.value)}><option>separate</option><option>direct</option></select></label></div><div className="lumina-actions"><span>Stored vision memories: {data?.vision_memories ?? 0}</span><span>{message}</span></div></SettingsSubpage>;
}

type RSSFeed = { feed_id: string; name: string; url: string; tags: string[]; active: boolean; success_count: number; fail_count: number; last_used?: number };

function RSSFeedsWindow() {
  const [feeds, setFeeds] = useState<RSSFeed[]>([]);
  const [name, setName] = useState('');
  const [url, setUrl] = useState('');
  const [tags, setTags] = useState('');
  const [message, setMessage] = useState('');
  const [busy, setBusy] = useState(false);
  const read = async () => {
    try { const response = await checked(await fetch('/api/interface/rss-feeds')); setFeeds((await response.json()).feeds || []); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'RSS feeds unavailable.'); }
  };
  useEffect(() => { void read(); }, []);
  const mutate = async (request: Promise<Response>, success: string) => {
    setBusy(true);
    try { await checked(await request); setMessage(success); await read(); }
    catch (error) { setMessage(error instanceof Error ? error.message : 'RSS operation failed.'); }
    finally { setBusy(false); }
  };
  const add = async () => {
    if (!name.trim() || !url.trim()) { setMessage('Name and URL are required.'); return; }
    await mutate(fetch('/api/interface/rss-feeds', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim(), url: url.trim(), tags: tags.split(',').map(item => item.trim()).filter(Boolean) }) }), `Added ${name.trim()}.`);
    setName(''); setUrl(''); setTags('');
  };
  return <main className="rss-window"><header className="rss-window-heading"><div><span className="eyebrow">LUMINA / RESEARCH</span><h1><Rss size={22}/> RSS Feed Management</h1><p>Manage the feeds used by Lumina's research pipeline. Disabled feeds are never queried; tags help route domain-relevant questions.</p></div><IconButton label="Close RSS feed management" onClick={() => window.close()}><X size={18}/></IconButton></header><section className="rss-feed-list"><div className="rss-section-heading"><h2>Configured feeds</h2><span>{feeds.length} feeds</span></div>{feeds.length ? feeds.map(feed => { const total = feed.success_count + feed.fail_count; const rate = total ? `${Math.round((feed.success_count / total) * 100)}% (${feed.success_count}/${total})` : 'No track record yet'; return <article className={`rss-feed-card ${feed.active ? 'active' : 'inactive'}`} key={feed.feed_id}><div className="rss-feed-main"><label className="rss-toggle"><input type="checkbox" checked={feed.active} disabled={busy} onChange={e => void mutate(fetch(`/api/interface/rss-feeds/${encodeURIComponent(feed.feed_id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ active: e.target.checked }) }), `${feed.name} ${e.target.checked ? 'activated' : 'deactivated'}.`)}/><span/></label><div className="rss-feed-info"><h3>{feed.name}</h3><code>{feed.url}</code></div><span className={`rss-rate ${total ? 'tracked' : ''}`}>{rate}</span><IconButton label={`Reset statistics for ${feed.name}`} disabled={busy} onClick={() => void mutate(fetch(`/api/interface/rss-feeds/${encodeURIComponent(feed.feed_id)}/reset`, { method: 'POST' }), `Statistics reset for ${feed.name}.`)}><RefreshCw size={15}/></IconButton><IconButton label={`Remove ${feed.name}`} disabled={busy} onClick={() => void mutate(fetch(`/api/interface/rss-feeds/${encodeURIComponent(feed.feed_id)}`, { method: 'DELETE' }), `Removed ${feed.name}.`)}><X size={16}/></IconButton></div><label className="rss-tags"><span>Tags</span><input defaultValue={feed.tags.join(', ')} disabled={busy} onBlur={e => { const next = e.currentTarget.value.split(',').map(item => item.trim()).filter(Boolean); if (next.join(',') !== feed.tags.join(',')) void mutate(fetch(`/api/interface/rss-feeds/${encodeURIComponent(feed.feed_id)}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ tags: next }) }), `Tags updated for ${feed.name}.`); }}/></label></article>; }) : <p className="muted">No RSS feeds configured.</p>}</section><section className="rss-add"><div className="rss-section-heading"><h2>Add a new feed</h2></div><div className="rss-add-form"><label><span>Name</span><input value={name} onChange={e => setName(e.target.value)} placeholder="e.g. arXiv CS.AI"/></label><label className="rss-url"><span>RSS URL</span><input value={url} onChange={e => setUrl(e.target.value)} placeholder="https://example.org/feed.xml"/></label><label><span>Tags</span><input value={tags} onChange={e => setTags(e.target.value)} placeholder="science, technology"/></label><button disabled={busy} onClick={() => void add()}><Check size={15}/>Add feed</button></div><p className="rss-examples">Examples: arXiv CS.AI, arXiv Physics, Nature News.</p></section><p className="rss-message" role="status">{message}</p></main>;
}

function VisionLegacyPanel() {
  const [frame, setFrame] = useState<any>(null); const [faces, setFaces] = useState<any[]>([]); const [memories, setMemories] = useState<any[]>([]); const [analysis, setAnalysis] = useState('No analysis yet'); const [query, setQuery] = useState(''); const [name, setName] = useState(''); const [message, setMessage] = useState('');
  const read = () => void Promise.all([fetch('/api/interface/camera/frame').then(r => r.json()), fetch('/api/interface/vision/faces').then(r => r.json()), fetch(`/api/interface/vision/memories?query=${encodeURIComponent(query)}`).then(r => r.json())]).then(([camera, faceData, memoryData]) => { setFrame(camera); setFaces(faceData.faces || []); setMemories(memoryData.memories || []); }).catch(() => {});
  useEffect(() => { read(); const timer = setInterval(read, 700); return () => clearInterval(timer); }, [query]);
  const action = async (url: string, options?: RequestInit) => { try { const r = await checked(await fetch(url, options)); return await r.json(); } catch (e) { setMessage(e instanceof Error ? e.message : 'Vision action failed.'); return null; } };
  const analyze = async () => { const result = await action('/api/interface/vision/analyze', { method: 'POST' }); if (result) { setAnalysis(result.analysis || 'No analysis returned.'); setFaces(result.faces || faces); } };
  const capture = async () => { if (!name.trim()) return; const result = await action('/api/interface/vision/capture-face', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) }); if (result) { setName(''); setMessage(result.message); read(); } };
  return <div className="vision-legacy-panel"><div className="vision-control-bar"><button onClick={() => void action('/api/interface/camera/toggle', { method: 'POST' }).then(read)}><Eye size={14}/> {frame?.active ? 'Disable camera' : 'Enable camera'}</button><button onClick={() => void action('/api/interface/vision/analyze', { method: 'POST' }).then(result => result && setAnalysis(result.analysis || 'No analysis returned.'))}><Activity size={14}/>Capture & analyze</button><button onClick={() => void action('/api/interface/vision/reset', { method: 'POST' }).then(read)}><RefreshCw size={14}/>Reset camera</button></div><div className="vision-feed-frame">{frame?.frame ? <img src={`data:image/jpeg;base64,${frame.frame}`} alt="Live camera feed"/> : <div className="vision-feed-empty"><Eye size={24}/><span>Camera feed is off.</span></div>}{(frame?.faces || []).map((face: any, index: number) => { const loc = face.location || {}; return <div className="vision-feed-face" key={index} style={{ left: `${(loc.left || 0) / 6.4}%`, top: `${(loc.top || 0) / 4.8}%`, width: `${((loc.right || 0) - (loc.left || 0)) / 6.4}%`, height: `${((loc.bottom || 0) - (loc.top || 0)) / 4.8}%` }}><b>{face.name} ({Math.round((face.confidence || 0) * 100)}%)</b></div>; })}</div><details className="vision-legacy-section" open><summary><Activity size={14}/>Latest analysis</summary><p className="vision-analysis">{analysis}</p></details><details className="vision-legacy-section" open><summary><Eye size={14}/>Face recognition <span>{faces.length} known</span></summary><div className="face-list">{faces.map(face => <div key={face.id}><span>{face.name}</span><small>seen {face.seen_count || 0} times</small></div>)}</div><div className="vision-capture-row"><input placeholder="Register new person" value={name} onChange={e => setName(e.target.value)}/><button disabled={!name.trim() || !frame?.active} onClick={() => void capture()}><Eye size={14}/>Capture face</button></div></details><details className="vision-legacy-section" open><summary><Database size={14}/>Vision memory <span>{memories.length} shown</span></summary><div className="vision-search"><input placeholder="Search memories..." value={query} onChange={e => setQuery(e.target.value)}/><Search size={15}/></div>{memories.map((memory: any, index: number) => <p className="vision-memory" key={index}>{memory.analysis || JSON.stringify(memory)}</p>)}</details>{message && <p className="voice-message">{message}</p>}</div>;
}

function CameraToggle() {
  const [active, setActive] = useState(false); const [busy, setBusy] = useState(false);
  useEffect(() => { let live = true; const read = () => void fetch('/api/interface/camera/frame').then(r => r.json()).then(v => live && setActive(Boolean(v.active))).catch(() => {}); read(); const timer = setInterval(read, 1200); return () => { live = false; clearInterval(timer); }; }, []);
  const toggle = async () => { setBusy(true); try { const response = await checked(await fetch('/api/interface/camera/toggle', { method: 'POST' })); setActive(Boolean((await response.json()).active)); } finally { setBusy(false); } };
  return <label className="camera-toggle"><span>Camera</span><input type="checkbox" checked={active} disabled={busy} onChange={() => void toggle()}/><b>{active ? 'Live' : 'Off'}</b></label>;
}

function SettingsVoiceRecording() {
  const [recording, setRecording] = useState(false); const [seconds, setSeconds] = useState(0); const [message, setMessage] = useState(''); const recorder = useRef<MediaRecorder | null>(null); const chunks = useRef<Blob[]>([]);
  useEffect(() => { if (!recording) return; const timer = setInterval(() => setSeconds(value => value + 1), 1000); return () => clearInterval(timer); }, [recording]);
  const stop = () => { recorder.current?.stop(); setRecording(false); };
  const start = async () => { try { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); chunks.current = []; const media = new MediaRecorder(stream, { mimeType: 'audio/webm' }); recorder.current = media; media.ondataavailable = event => { if (event.data.size) chunks.current.push(event.data); }; media.onstop = async () => { stream.getTracks().forEach(track => track.stop()); const body = new FormData(); body.append('file', new Blob(chunks.current, { type: 'audio/webm' }), 'coqui-recording.webm'); setMessage('Uploading voice sample...'); try { await checked(await fetch('/api/interface/voice/coqui-recording', { method: 'POST', body })); setMessage('Voice sample saved and loaded into Coqui.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Upload failed.'); } }; setSeconds(0); media.start(250); setRecording(true); setMessage('Recording... read the sample naturally for 15–30 seconds.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Microphone permission is unavailable.'); } };
  return <SettingsSubpage tab="voice"><div className="settings-section-title">Coqui voice cloning</div><div className="conditional-panel coqui-panel"><h4>Record your voice</h4><p>Record a single speaker for 15–30 seconds. Stop when finished; the sample is converted and loaded automatically.</p><div className="coqui-record-controls"><button className={recording ? 'recording' : ''} onClick={() => recording ? stop() : void start()}>{recording ? <Square size={16}/> : <Mic size={16}/>} {recording ? 'Stop recording' : 'Start recording'}</button><strong>{String(Math.floor(seconds / 60)).padStart(2, '0')}:{String(seconds % 60).padStart(2, '0')}</strong></div><div className="coqui-script"><b>Read this aloud</b><p>The quick brown fox jumps over the lazy dog near the riverbank. Every morning, I wake up and look out the window at the sky. Sometimes it is bright and clear; other times, clouds gather on the horizon. I enjoy a warm cup of coffee while thinking about the day ahead.</p></div><p className="voice-message">{message}</p></div><div className="settings-section-title">Current voice</div><div className="settings-object-grid"><div><span>Provider</span><b>Coqui</b></div><div><span>Reference</span><b>recorded_voice.wav</b></div></div></SettingsSubpage>;
}

function SettingsVoiceComplete() {
  const [values, setValues] = useState<Record<string, any>>({}); const [recording, setRecording] = useState(false); const [seconds, setSeconds] = useState(0); const [message, setMessage] = useState(''); const recorder = useRef<MediaRecorder | null>(null); const chunks = useRef<Blob[]>([]);
  useEffect(() => { void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => setValues(r.values || {})); }, []);
  useEffect(() => { if (!recording) return; const timer = setInterval(() => setSeconds(value => value + 1), 1000); return () => clearInterval(timer); }, [recording]);
  const set = (key: string, value: any) => setValues(v => ({ ...v, [key]: value }));
  const save = async () => { try { const response = await checked(await fetch('/api/interface/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) })); const result = await response.json(); setMessage(result.active_tts ? `Voice settings saved. Active TTS: ${result.active_tts}.` : 'Voice settings saved.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Save failed.'); } };
  const upload = async (file: File) => { const body = new FormData(); body.append('file', file, file.name); setMessage('Uploading voice sample...'); try { const response = await checked(await fetch('/api/interface/voice/coqui-recording', { method: 'POST', body })); const result = await response.json(); set('COQUI_VOICE_REFERENCE', result.path || 'data/voices/recorded_voice.wav'); setMessage(result.active_provider === 'coqui' ? 'WAV sample saved and active in Coqui.' : `WAV saved, but active TTS is ${result.active_provider || 'unavailable'}.`); } catch (e) { setMessage(e instanceof Error ? e.message : 'WAV upload failed.'); } };
  const stop = () => { recorder.current?.stop(); setRecording(false); };
  const start = async () => { try { const stream = await navigator.mediaDevices.getUserMedia({ audio: true }); chunks.current = []; const media = new MediaRecorder(stream, { mimeType: 'audio/webm' }); recorder.current = media; media.ondataavailable = event => { if (event.data.size) chunks.current.push(event.data); }; media.onstop = () => { stream.getTracks().forEach(track => track.stop()); void upload(new File([new Blob(chunks.current, { type: 'audio/webm' })], 'coqui-recording.webm')); }; setSeconds(0); media.start(250); setRecording(true); setMessage('Recording...'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Microphone permission is unavailable.'); } };
  const test = async () => { try { const r = await checked(await fetch('/api/interface/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: 'This is a test of my current voice.' }) })); const audio = new Audio(URL.createObjectURL(await r.blob())); void audio.play(); setMessage('Voice test started.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Voice test failed.'); } };
  const nums = [['BARGE_IN_SENSITIVITY','Barge-in sensitivity'],['VAD_AGGRESSIVENESS','VAD aggressiveness'],['VAD_ONSET_CHUNKS','Onset chunks'],['VAD_SILENCE_DURATION','Silence duration'],['VAD_MIN_SPEECH_DURATION','Minimum speech'],['VAD_ENERGY_GATE_FACTOR','Energy gate'],['INTER_SENTENCE_PAUSE_MS','Sentence pause'],['TTS_POST_ROLL_MS','Echo post-roll']];
  return <SettingsSubpage tab="voice"><div className="settings-section-title">TTS and STT engines</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>TTS provider</span><select value={values.TTS_PROVIDER || ''} onChange={e => set('TTS_PROVIDER', e.target.value)}><option value="pyttsx3">pyttsx3 - Windows voice</option><option value="coqui">Coqui XTTS voice clone</option><option value="kokoro">Kokoro</option><option value="edge-tts">Edge neural voice</option><option value="elevenlabs">ElevenLabs</option></select></label><label className="settings-field"><span>STT provider</span><select value={values.STT_PROVIDER || ''} onChange={e => set('STT_PROVIDER', e.target.value)}><option>whisper</option><option>faster_whisper</option><option>openai</option></select></label><label className="settings-field"><span>Whisper model</span><select value={values.WHISPER_MODEL || 'base'} onChange={e => set('WHISPER_MODEL', e.target.value)}>{['tiny','base','small','medium','large'].map(v => <option key={v}>{v}</option>)}</select></label><label className="settings-field"><span>Voice language</span><select value={values.VOICE_LANGUAGE || 'en'} onChange={e => set('VOICE_LANGUAGE', e.target.value)}>{voiceLanguages.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label><label className="settings-field"><span>Response language</span><select value={values.RESPONSE_LANGUAGE || 'auto'} onChange={e => set('RESPONSE_LANGUAGE', e.target.value)}><option value="auto">Auto - follow conversation</option>{voiceLanguages.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label></div>{values.TTS_PROVIDER === 'coqui' && <div className="conditional-panel coqui-panel"><h4>Coqui voice cloning</h4><label className="settings-field"><span>Current WAV reference</span><input value={values.COQUI_VOICE_REFERENCE || ''} onChange={e => set('COQUI_VOICE_REFERENCE', e.target.value)}/></label><label className="file-picker"><span>Select WAV voice sample</span><input type="file" accept="audio/wav,audio/x-wav,audio/*" onChange={e => { const file = e.target.files?.[0]; if (file) void upload(file); }}/></label><div className="coqui-record-controls"><button className={recording ? 'recording' : ''} onClick={() => recording ? stop() : void start()}>{recording ? <Square size={16}/> : <Mic size={16}/>} {recording ? 'Stop recording' : 'Start recording'}</button><strong>{String(Math.floor(seconds / 60)).padStart(2, '0')}:{String(seconds % 60).padStart(2, '0')}</strong><button onClick={() => void test()}><Volume2 size={16}/>Test voice</button></div><p>Read the sample naturally for 15–30 seconds, then stop. The recording is converted and loaded automatically.</p></div>}<div className="settings-section-title">Always-listen and echo control</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>Sentence streaming</span><input type="checkbox" checked={Boolean(values.TTS_STREAMING)} onChange={e => set('TTS_STREAMING', e.target.checked)}/></label><label className="settings-field"><span>Barge-in interruption</span><input type="checkbox" checked={Boolean(values.TTS_BARGE_IN)} onChange={e => set('TTS_BARGE_IN', e.target.checked)}/></label>{nums.map(([key,label]) => <label className="settings-field" key={key}><span>{label}</span><input type="number" value={values[key] ?? ''} onChange={e => set(key, Number(e.target.value))}/></label>)}</div><div className="lumina-actions"><button onClick={() => void save()}><Check size={15}/>Save voice settings</button><span>{message}</span></div></SettingsSubpage>;
}

function SettingsAnalyticsLegacy() {
  const [data, setData] = useState<any>(null); const [analysis, setAnalysis] = useState(''); const [generated, setGenerated] = useState(''); const [busy, setBusy] = useState(false);
  const read = () => void fetch('/api/interface/settings/analytics').then(checked).then(r => r.json()).then(setData).catch(() => {});
  useEffect(() => { read(); const timer = setInterval(read, 6000); return () => clearInterval(timer); }, []);
  const analyze = async () => { setBusy(true); try { const r = await checked(await fetch('/api/interface/settings/analytics/self-analysis', { method: 'POST' })); setAnalysis((await r.json()).analysis || ''); setGenerated(new Date().toLocaleString()); } catch (e) { setAnalysis(e instanceof Error ? e.message : 'Analysis unavailable.'); } finally { setBusy(false); } };
  const capabilities = Object.entries(data?.capabilities || {});
  return <SettingsSubpage tab="analytics"><div className="analytics-legacy-intro"><h4>LOG ANALYTICS & SELF-ANALYSIS</h4><p>Lumina analyzes her own cognitive trends and log patterns. Trend data comes from the Observatory history (last 200 ticks). Self-analysis uses the LLM to interpret what the numbers mean.</p></div><div className="settings-section-title">Cognitive Metric Trends / Observatory History</div>{data && !data.history?.length ? <p className="muted">Only 0 ticks collected — need at least 3. Keep chatting!</p> : <div className="analytics-metrics">{Object.entries(data?.trends || {}).map(([key, item]: [string, any]) => <div key={key}><span>{key.toUpperCase()}</span><b>{Number(item.current).toFixed(3)}</b><em className={item.delta >= 0 ? 'trend-up' : 'trend-down'}>{item.delta >= 0 ? '↑' : '↓'} {Math.abs(item.delta).toFixed(3)} · avg {Number(item.average).toFixed(3)}</em><small>range {Number(item.minimum).toFixed(3)}–{Number(item.maximum).toFixed(3)}</small></div>)}</div>}<div className="settings-section-title">Recent log analysis / last 500 lines</div><div className="settings-stat-grid"><div className="settings-stat"><span>Info</span><strong>{data?.log?.infos ?? 0}</strong></div><div className="settings-stat"><span>Warnings</span><strong>{data?.log?.warnings ?? 0}</strong></div><div className="settings-stat"><span>Errors</span><strong className={data?.log?.errors ? 'value-warn' : ''}>{data?.log?.errors ?? 0}</strong></div></div>{data?.log?.recent_errors?.length ? <pre className="settings-log">{data.log.recent_errors.join('\n')}</pre> : <p className="settings-ok-line">No errors in the current log window.</p>}<div className="settings-section-title">Lumina Self-Analysis / AI-powered cognitive self-analysis</div><p className="muted">Lumina reads her own metrics and writes a free-form analysis of trends, anomalies, and cognitive health. Requires LLM.</p><div className="settings-object-grid">{capabilities.length ? capabilities.map(([name, value]: [string, any]) => <div key={name}><span>Capability {name}</span><b>{typeof value === 'object' ? `${value.confidence ?? 'n/a'} · trend ${value.trend ?? '0'}` : String(value)}</b></div>) : <div><span>Capabilities</span><b>No capability records</b></div>}</div><button className="settings-action-button" disabled={busy} onClick={() => void analyze()}><Sparkles size={15}/>{busy ? 'Analyzing...' : 'Generate self-analysis'}</button>{analysis && <div className="settings-analysis"><b>Generated {generated}</b><p>{analysis}</p></div>}</SettingsSubpage>;
}

function SettingsVisionFull() {
  const [values, setValues] = useState<Record<string, any>>({});
  const [vision, setVision] = useState<any>(null);
  const [faces, setFaces] = useState<any[]>([]);
  const [name, setName] = useState('');
  const [message, setMessage] = useState('');
  const readFaces = () => void fetch('/api/interface/vision/faces').then(checked).then(r => r.json()).then(r => setFaces(r.faces || [])).catch(() => {});
  const read = () => void Promise.all([
    fetch('/api/interface/settings/tools').then(checked).then(r => r.json()),
    fetch('/api/interface/vision/faces').then(checked).then(r => r.json()),
  ]).then(([toolsData, faceData]) => { setVision(toolsData.vision); setFaces(faceData.faces || []); }).catch(() => {});
  useEffect(() => { void fetch('/api/interface/settings').then(checked).then(r => r.json()).then(r => setValues(r.values || {})); read(); }, []);
  const set = (key: string, value: any) => setValues(current => ({ ...current, [key]: value }));
  const save = async () => { try { await checked(await fetch('/api/interface/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) })); setMessage('Saved. Restart recommended for camera configuration changes.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Save failed.'); } };
  const applyVision = async (face?: boolean) => { try { const r = await checked(await fetch('/api/interface/vision/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ camera_id: Number(values.CAMERA_ID ?? 0), fps: Number(values.CAMERA_FPS ?? 5), resolution: values.CAMERA_RESOLUTION || '640x480', face_detection: face }) })); setVision(await r.json()); readFaces(); setMessage('Vision settings applied.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Camera settings failed.'); } };
  const cameraTest = async () => { try { const r = await checked(await fetch('/api/interface/camera/toggle', { method: 'POST' })); const result = await r.json(); setVision((current: any) => ({ ...current, camera_active: result.active })); readFaces(); setMessage('Camera test completed.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Camera test failed.'); } };
  const captureFace = async () => { if (!name.trim()) return; try { const r = await checked(await fetch('/api/interface/vision/capture-face', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name: name.trim() }) })); const result = await r.json(); setFaces(result.faces || []); setName(''); setMessage(result.message || 'Face captured.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Face capture failed.'); } };
  return <SettingsSubpage tab="vision"><div className="settings-section-title">Camera capture</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>Camera selection</span><select value={values.CAMERA_ID ?? 0} onChange={e => set('CAMERA_ID', Number(e.target.value))}><option value="0">Camera 0</option><option value="1">Camera 1</option><option value="2">Camera 2</option></select></label><label className="settings-field"><span>Frame rate</span><select value={values.CAMERA_FPS ?? 5} onChange={e => set('CAMERA_FPS', Number(e.target.value))}><option value="5">5 fps</option><option value="10">10 fps</option><option value="15">15 fps</option></select></label><label className="settings-field"><span>Resolution</span><select value={values.CAMERA_RESOLUTION || '640x480'} onChange={e => set('CAMERA_RESOLUTION', e.target.value)}><option>640x480</option><option>1280x720</option></select></label><label className="settings-field"><span>Startup camera</span><input type="checkbox" checked={Boolean(values.CAMERA_AUTOSTART)} onChange={e => set('CAMERA_AUTOSTART', e.target.checked)}/></label></div><div className="vision-live-strip"><span className={vision?.camera_active ? 'value-ok' : ''}>Camera {vision?.camera_active ? 'active' : 'off'}</span><span>{vision?.model || values.LAVA_MODEL || 'No model'}</span><span>{vision?.resolution || values.CAMERA_RESOLUTION || '640x480'}</span></div><div className="lumina-actions"><button onClick={() => void applyVision()}><Check size={15}/>Apply capture settings</button><button onClick={() => void cameraTest()}><Eye size={15}/>Test camera</button></div><div className="settings-section-title">Face recognition</div><div className="conditional-panel"><label className="settings-field"><span>Enable face recognition</span><input type="checkbox" checked={Boolean(vision?.face_detection)} onChange={e => void applyVision(e.target.checked)}/></label><div className="settings-object-grid"><div><span>Known faces</span><b>{faces.length || vision?.known_faces || 0}</b></div><div><span>Recognition engine</span><b>{vision?.initialized ? 'Initialized' : 'Unavailable'}</b></div></div><div className="face-list">{faces.length ? faces.map(face => <div key={face.id || face.name}><span>{face.name}</span><small>seen {face.seen_count || 0} times</small></div>) : <p className="muted">No recorded faces.</p>}</div><div className="vision-capture-row"><input placeholder="Register new person from current camera" value={name} onChange={e => setName(e.target.value)} onKeyDown={e => { if (e.key === 'Enter') void captureFace(); }}/><button disabled={!name.trim() || !vision?.camera_active} onClick={() => void captureFace()}><Eye size={15}/>Capture face</button></div></div><div className="settings-section-title">Vision memory and model</div><div className="settings-form settings-form-compact"><label className="settings-field"><span>LAVA model</span><input value={values.LAVA_MODEL || ''} onChange={e => set('LAVA_MODEL', e.target.value)}/></label><label className="settings-field"><span>LLM routing</span><select value={values.VISION_LLM_MODE || 'separate'} onChange={e => set('VISION_LLM_MODE', e.target.value)}><option>separate</option><option>direct</option></select></label><label className="settings-field"><span>Ambient interval</span><input type="number" value={values.AMBIENT_VISION_INTERVAL ?? 90} onChange={e => set('AMBIENT_VISION_INTERVAL', Number(e.target.value))}/></label></div><p className="muted">Stored vision memories: {vision?.vision_memories ?? 0}</p><div className="lumina-actions"><button onClick={() => void save()}><Check size={15}/>Save vision settings</button><button onClick={() => void fetch('/api/interface/vision/clear-memory', { method: 'POST' }).then(read)}><Database size={15}/>Clear vision memory</button><span>{message}</span></div></SettingsSubpage>;
}

function SettingsVisionSurface() {
  return <><CameraToggle/><VisionFeed data={null}/><SettingsVisionFull/></>;
}

function CapabilityExperimentPanel({ proposalId }: { proposalId: string }) {
  const [state, setState] = useState<any>(null);
  useEffect(() => { let live = true; const read = () => void fetch(`/api/interface/capability-proposals/${proposalId}/experiment`).then(checked).then(r => r.json()).then(value => { if (live) setState(value); }).catch(() => {}); read(); const timer = window.setInterval(read, 5000); return () => { live = false; window.clearInterval(timer); }; }, [proposalId]);
  const experiment = state?.experiment;
  if (!experiment) return <div className="settings-analysis"><b>Experiment not initialized.</b><p className="muted">Set the proposal status to in_progress to start the bounded Workspace experiment.</p></div>;
  const baselineCount = experiment.baseline?.length || 0; const trialCount = experiment.trials?.length || 0; const stabilizationCount = experiment.stabilization?.length || 0; const stabilizationWindows = experiment.stabilization_windows?.length || 0;
  return <div className="development-experiment"><div className="settings-object-grid"><div><span>Experiment state</span><b>{experiment.status}</b></div><div><span>Baseline</span><b>{baselineCount}/{experiment.max_baseline_steps}</b></div><div><span>Trials</span><b>{trialCount}/{experiment.max_trial_steps}</b></div><div><span>Stabilization</span><b>{stabilizationWindows}/{experiment.max_stabilization_windows || 3} windows · {stabilizationCount}/{experiment.max_stabilization_window_steps || 6}</b></div><div><span>Subsystem</span><b>{experiment.subsystem}</b></div></div>{experiment.result && <div className="settings-analysis"><b>Evidence</b><p>Baseline {experiment.result.baseline_score} → trials {experiment.result.trial_score} · improvement {experiment.result.improvement}</p><p>{experiment.result.confirmation === 'pending_long_run' ? 'Initial improvement recorded; long-run confirmation is in progress.' : experiment.result.evidence}</p></div>}{state?.verified_capability && <div className="settings-analysis"><b>Verified capability</b><p>{state.verified_capability.name} · {state.verified_capability.status}</p></div>}</div>;
}

function SettingsDevelopmentLegacy() {
  const empty = { title: '', description: '', category: 'cognitive', motivation: '', expected_capability: '', constraints: '', success_criteria: '', priority: 3 };
  const [form, setForm] = useState<Record<string, any>>(empty); const [items, setItems] = useState<any[]>([]); const [selected, setSelected] = useState<any>(null); const [message, setMessage] = useState(''); const [analyzing, setAnalyzing] = useState(false);
  const read = () => void fetch('/api/interface/capability-proposals').then(checked).then(r => r.json()).then(r => setItems(r.proposals || [])).catch(e => setMessage(e instanceof Error ? e.message : 'Development proposals unavailable.'));
  useEffect(() => { read(); }, []);
  const set = (key: string, value: any) => setForm(current => ({ ...current, [key]: value }));
  const create = async () => { if (!form.title.trim() || !form.description.trim()) { setMessage('Add a title and a description first.'); return; } try { const response = await checked(await fetch('/api/interface/capability-proposals', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(form) })); const item = await response.json(); setItems(current => [item, ...current]); setSelected(item); setForm(empty); setMessage('Proposal saved and development prompt generated.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Proposal could not be saved.'); } };
  const askLuminaToPropose = async () => { setAnalyzing(true); setMessage('Lumina is considering a new capability...'); try { const response = await checked(await fetch('/api/interface/capability-proposals/generate', { method: 'POST' })); const item = await response.json(); setItems(current => [item, ...current]); setSelected(item); setMessage('Lumina created a draft proposal for review.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Lumina proposal generation failed.'); } finally { setAnalyzing(false); } };
  const update = async (item: any, status: string) => { try { const response = await checked(await fetch(`/api/interface/capability-proposals/${item.id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ status }) })); const next = await response.json(); setItems(current => current.map(entry => entry.id === next.id ? next : entry)); setSelected(next); } catch (e) { setMessage(e instanceof Error ? e.message : 'Proposal update failed.'); } };
  const askOrganism = async () => { if (!selected) return; setAnalyzing(true); setMessage('Submitting proposal to Lumina...'); try { const response = await checked(await fetch(`/api/interface/capability-proposals/${selected.id}/analyze`, { method: 'POST' })); const next = await response.json(); setItems(current => current.map(entry => entry.id === next.id ? next : entry)); setSelected(next); setMessage('Lumina analysis received and attached to the proposal.'); } catch (e) { setMessage(e instanceof Error ? e.message : 'Lumina analysis failed.'); } finally { setAnalyzing(false); } };
  const copy = async () => { if (selected?.generated_prompt) { await navigator.clipboard?.writeText(selected.generated_prompt); setMessage('Development prompt copied.'); } };
  return <SettingsSubpage tab="development"><div className="settings-section-title">Capability incubator</div><p className="muted">Propose new cognitive abilities, external tools, sensors or embodiment ideas. The organism turns each proposal into an implementation brief for review.</p><div className="development-form"><label className="settings-field"><span>Proposal title</span><input value={form.title} onChange={e => set('title', e.target.value)} placeholder="e.g. Posture-aware interaction"/></label><label className="settings-field"><span>Category</span><select value={form.category} onChange={e => set('category', e.target.value)}>{['cognitive','tool','sensor','embodiment','learning','interaction','experiment'].map(option => <option key={option}>{option}</option>)}</select></label><label className="settings-field wide"><span>What should be developed?</span><textarea value={form.description} onChange={e => set('description', e.target.value)} placeholder="Describe the capability, tool or physical integration."/></label><label className="settings-field"><span>Why does it matter?</span><textarea value={form.motivation} onChange={e => set('motivation', e.target.value)}/></label><label className="settings-field"><span>Expected capability</span><textarea value={form.expected_capability} onChange={e => set('expected_capability', e.target.value)}/></label><label className="settings-field"><span>Constraints and hardware</span><textarea value={form.constraints} onChange={e => set('constraints', e.target.value)} placeholder="Optional hardware, privacy or latency constraints."/></label><label className="settings-field"><span>Evidence of success</span><textarea value={form.success_criteria} onChange={e => set('success_criteria', e.target.value)} placeholder="How should we verify it?"/></label><label className="settings-field"><span>Priority: {form.priority}/5</span><input type="range" min="1" max="5" value={form.priority} onChange={e => set('priority', Number(e.target.value))}/></label></div><div className="lumina-actions"><button onClick={() => void create()}><Lightbulb size={15}/>Create proposal</button><button disabled={analyzing} onClick={() => void askLuminaToPropose()}><Sparkles size={15}/>{analyzing ? 'Lumina is thinking...' : 'Ask Lumina to propose'}</button><span>{message}</span></div><div className="settings-section-title">Proposal queue</div><div className="development-queue">{items.length ? items.map(item => <button className={`development-item ${selected?.id === item.id ? 'active' : ''}`} key={item.id} onClick={() => setSelected(item)}><span><b>{item.title}</b><small>{item.source} · {item.category} · priority {item.priority} · {item.status}</small></span><ChevronRight size={15}/></button>) : <p className="muted">No proposals yet.</p>}</div>{selected && <section className="development-detail"><div className="development-detail-heading"><div><h4>{selected.title}</h4><p>{selected.analysis}</p></div><select value={selected.status} onChange={e => void update(selected, e.target.value)}>{['draft','accepted','in_progress','completed','deferred','rejected'].map(status => <option key={status}>{status}</option>)}</select></div><div className="settings-object-grid">{(selected.affected_subsystems || []).map((name: string) => <div key={name}><span>Affected subsystem</span><b>{name}</b></div>)}</div><div className="settings-section-title">Organism analysis</div>{selected.organism_analysis ? <div className="settings-analysis">{selected.organism_analysis}</div> : <p className="muted">Not submitted to Lumina yet.</p>}<div className="development-actions"><button className="settings-action-button" disabled={analyzing} onClick={() => void askOrganism()}><Sparkles size={15}/>{analyzing ? 'Analyzing...' : 'Ask Lumina to analyze'}</button><button className="settings-action-button" onClick={() => void copy()}><Copy size={15}/>Copy development prompt</button></div><div className="settings-section-title">Development prompt</div><pre className="settings-log development-prompt">{selected.generated_prompt}</pre></section>}</SettingsSubpage>;
}

function CapabilityExperimentOverview() {
  const [proposal, setProposal] = useState<any>(null);
  useEffect(() => { let live = true; const read = () => void fetch('/api/interface/capability-proposals').then(checked).then(r => r.json()).then(r => { if (live) setProposal((r.proposals || []).find((item: any) => item.status === 'in_progress' && item.category === 'experiment') || null); }).catch(() => {}); read(); const timer = window.setInterval(read, 5000); return () => { live = false; window.clearInterval(timer); }; }, []);
  return proposal ? <section className="development-detail"><div className="settings-section-title">Active experiment</div><CapabilityExperimentPanel proposalId={proposal.id}/></section> : null;
}

function SettingsDevelopment() {
  return <><SettingsDevelopmentLegacy/><CapabilityExperimentOverview/></>;
}

function SettingsView() {
  const requested = new URLSearchParams(window.location.search).get('tab') as SettingsTab | null;
  const [tab, setTab] = useState<SettingsTab>(requested && settingTabs.some(item => item.id === requested) ? requested : 'llm');
  const [values, setValues] = useState<Record<string, SettingsValue>>({});
  const [secrets, setSecrets] = useState<string[]>([]);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');
  useEffect(() => { void fetch('/api/interface/settings').then(checked).then(async response => { const result = await response.json(); setValues(result.values || {}); setSecrets(result.secret_fields || []); }).catch(e => setError(e instanceof Error ? e.message : 'Settings unavailable.')); }, []);
  async function save() { setMessage('Saving...'); setError(''); try { const response = await checked(await fetch('/api/interface/settings', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ values }) })); const result = await response.json(); setValues(result.values || values); setMessage('Saved. Restart recommended for provider, memory, or vision changes.'); } catch (e) { setMessage(''); setError(e instanceof Error ? e.message : 'Settings could not be saved.'); } }
  const fields = settingsFields[tab as keyof typeof settingsFields] || [];
  const extraView = ({ status: <SettingsStatus />, lumina: <SettingsLumina />, tools: <SettingsToolsFull />, analytics: <SettingsAnalyticsLegacy />, llm: <SettingsLlmFull />, memory: <SettingsMemoryComplete />, voice: <SettingsVoiceComplete />, vision: <SettingsVisionSurface />, connector: <SettingsUniversalConnector />, development: <SettingsDevelopment /> } as Partial<Record<SettingsTab, ReactNode>>)[tab];
  if (extraView) return extraView;
  return <main className="settings-view"><header className="settings-view-header"><div><span className="eyebrow">LUMINA / CONFIGURATION</span><h1>Settings</h1></div><div className="settings-view-actions"><span className="settings-message">{message || error}</span><button className="settings-save" onClick={() => void save()}><Check size={15}/>Save & Apply</button><IconButton label="Close settings window" onClick={() => window.close()}><X size={18}/></IconButton></div></header><div className="settings-layout"><nav className="settings-nav" aria-label="Settings sections">{settingTabs.map(item => <button className={tab === item.id ? 'active' : ''} key={item.id} onClick={() => setTab(item.id)}>{item.icon}<span>{item.label}</span></button>)}</nav><section className="settings-content"><div className="settings-intro"><span className="settings-tab-icon">{settingTabs.find(item => item.id === tab)?.icon}</span><div><h2>{settingTabs.find(item => item.id === tab)?.label}</h2><p>{tab === 'llm' ? 'Identity, provider routing, research backend, response budgets, and prompt shaping.' : tab === 'memory' ? 'Choose the memory substrate and persistent storage locations.' : tab === 'voice' ? 'Tune fluid listening, speech recognition, synthesis, and echo control.' : tab === 'vision' ? 'Configure camera capture, vision routing, and ambient perception.' : tab === 'connector' ? 'Connect local sensors and services through a controlled universal gateway.' : 'Live configuration surface for the Lumina cognitive runtime.'}</p></div></div>{extraView || (fields.length ? <div className="settings-form">{fields.map(field => <label className={`settings-field ${field.type === 'textarea' ? 'wide' : ''}`} key={field.key}><span>{field.label}</span>{field.type === 'textarea' ? <textarea value={String(values[field.key] ?? '')} onChange={e => setValues(current => ({ ...current, [field.key]: e.target.value }))}/> : field.type === 'checkbox' ? <input type="checkbox" checked={Boolean(values[field.key])} onChange={e => setValues(current => ({ ...current, [field.key]: e.target.checked }))}/> : field.type === 'select' ? <select value={String(values[field.key] ?? '')} onChange={e => setValues(current => ({ ...current, [field.key]: e.target.value }))}>{field.options?.map(option => <option value={option} key={option}>{settingOptionLabel(field.key, option)}</option>)}</select> : <input type={secrets.includes(field.key) ? 'password' : field.type === 'number' ? 'number' : 'text'} value={String(values[field.key] ?? '')} onChange={e => setValues(current => ({ ...current, [field.key]: field.type === 'number' ? Number(e.target.value) : e.target.value }))}/>}</label>)}</div> : <div className="settings-placeholder"><h3>{tab === 'status' ? 'Runtime status' : tab === 'lumina' ? 'Lumina mind' : tab === 'tools' ? 'Cognitive tools' : 'Analytics'}</h3><p>This surface remains connected to the legacy page while its detailed controls are migrated. Use the navigation links below for the complete legacy view.</p><a href={`/${tab === 'lumina' ? 'lumina' : tab === 'tools' ? 'settings' : tab === 'analytics' ? 'settings' : 'settings'}`} target="_blank" rel="noreferrer">Open legacy {tab} page <ChevronRight size={14}/></a></div>)}</section></div></main>;
}

async function checked(response: Response) {
  if (!response.ok) { const error = await response.json().catch(() => ({})); throw new Error(typeof error.detail === 'string' ? error.detail : `Request failed (${response.status}).`); }
  return response;
}

function savedMessages(): Message[] {
  try {
    const value = JSON.parse(localStorage.getItem(storageKey) || '[]');
    return Array.isArray(value) ? value.filter(m => m && typeof m.id === 'string' && !m.id.startsWith('presence-') && ['user', 'assistant'].includes(m.role) && typeof m.text === 'string').slice(-200) : [];
  } catch { return []; }
}

export default function App() {
  const view = new URLSearchParams(window.location.search).get('view');
  if (view === 'flux-dialogue') return <FluxDialogueWindow />;
  if (view === 'health') return <CognitiveHealthWindow />;
  if (view === 'rss') return <RSSFeedsWindow />;
  if (view === 'settings') return <SettingsView />;
  const [messages, setMessages] = useState<Message[]>(savedMessages);
  const [healthSnapshot, setHealthSnapshot] = useState<CognitiveHealthData | null>(null);
  const [showTelemetry, setShowTelemetry] = useState(() => localStorage.getItem(telemetryVisibilityKey) === 'true');
  const [draft, setDraft] = useState('');
  const [status, setStatus] = useState<Status | null>(null);
  const [personaName, setPersonaName] = useState('Lumina');
  const [networkState, setNetworkState] = useState<NetworkState>({ enabled: false, children: [] });
  const [voiceMemSpace, setVoiceMemSpace] = useState<VoiceMemSpaceData | null>(null);
  const [networkOpen, setNetworkOpen] = useState(false);
  const [networkBusy, setNetworkBusy] = useState(false);
  const [networkResult, setNetworkResult] = useState('');
  const [cameraMode, setCameraMode] = useState(false);
  const [cameraFrame, setCameraFrame] = useState<string | null>(null);
  const [cameraFaces, setCameraFaces] = useState<{ name: string; confidence?: number; location?: { left?: number; top?: number; right?: number; bottom?: number } }[]>([]);
  const [connected, setConnected] = useState(false);
  const [responseLanguage, setResponseLanguage] = useState('auto');
  const [webSearchMode, setWebSearchMode] = useState<'off' | 'auto' | 'always'>('off');
  const [responseVerbosity, setResponseVerbosity] = useState<'concise' | 'verbose'>('concise');
  const [inspector, setInspector] = useState(false);
  const [tab, setTab] = useState<typeof tabs[number]>('Overview');
  const [voice, setVoice] = useState(false);
  const [history, setHistory] = useState(false);
  const [settings, setSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [recording, setRecording] = useState(false);
  const [micEnabled, setMicEnabled] = useState(true);
  const [audioBusy, setAudioBusy] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const [level, setLevel] = useState(0);
  const [interimTranscript, setInterimTranscript] = useState('');
  const [reduced, setReduced] = useState(() => matchMedia('(prefers-reduced-motion: reduce)').matches);
  const [panelWidth, setPanelWidth] = useState(380);
  const [copied, setCopied] = useState('');
  const [atBottom, setAtBottom] = useState(true);
  const [eventId, setEventId] = useState<string | null>(null);
  const [organismHover, setOrganismHover] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const audioAbort = useRef<AbortController | null>(null);
  const playback = useRef<HTMLAudioElement | null>(null);
  const playbackUrl = useRef<string | null>(null);
  const transcript = useRef<HTMLDivElement>(null);
  const composer = useRef<HTMLTextAreaElement>(null);
  const inspectorTrigger = useRef<HTMLButtonElement>(null);
  const inspectorClose = useRef<HTMLButtonElement>(null);
  const mounted = useRef(true);
  const operation = useRef(0);
  const seenPresence = useRef(new Set<string>());
  const voiceWorking = useRef(false);
  const voiceGate = useRef(true);
  const voiceActive = useRef(false);
  const echoUntil = useRef(0);
  const speechQueue = useRef<string[]>([]);
  const speechDraining = useRef(false);
  const speechGeneration = useRef(0);
  const speakingRef = useRef(false);
  const audioBusyRef = useRef(false);
  const busyRef = useRef(false);
  const statusBusyRef = useRef(false);
  const submitVoice = useRef<(text: string, automatic?: boolean) => Promise<void>>(async () => {});
  submitVoice.current = send;
  voiceActive.current = voice && micEnabled;
  speakingRef.current = speaking;
  audioBusyRef.current = audioBusy;
  busyRef.current = busy;
  statusBusyRef.current = Boolean(status?.busy);
  voiceGate.current = busy || Boolean(status?.busy) || audioBusy || speaking || !connected || !status?.ready;

  useEffect(() => {
    if (!voice) setMicEnabled(true);
  }, [voice]);
  useEffect(() => {
    let stopped = false;
    const loadVoiceMemSpace = async () => {
      try {
        const response = await fetch('/api/interface/voicemem/space');
        if (response.ok && !stopped) setVoiceMemSpace(await response.json());
      } catch { /* optional subsystem; keep the main interface available */ }
    };
    void loadVoiceMemSpace();
    const timer = window.setInterval(loadVoiceMemSpace, 15000);
    return () => { stopped = true; window.clearInterval(timer); };
  }, []);
  useEffect(() => {
    if (!voice) {
      // All voice exit paths share this reset, including the rail and hangup
      // controls.  Do not leave the normal composer gated by stale audio UI.
      voiceWorking.current = false;
      abort.current?.abort();
      setBusy(false);
      setRecording(false);
      setAudioBusy(false);
    }
  }, [voice]);
  useEffect(() => {
    if (!voice || !micEnabled || !connected || !status?.ready) return;
    let closed = false;
    let stop: (() => Promise<void>) | undefined;
    const transcription = new AbortController();
    void listenContinuously({
      blocked: () => {
        const bargeIn = speakingRef.current || audioBusyRef.current;
        return closed || voiceWorking.current || performance.now() < echoUntil.current || (voiceGate.current && !bargeIn);
      },
      onLevel: value => { if (!closed) setLevel(value); },
      onSpeechStart: () => { if (!closed) setInterimTranscript(''); },
      onPartial: async blob => {
        if (closed) return;
        const response = await checked(await fetch('/api/interface/transcribe', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: blob, signal: transcription.signal }));
        const { text } = await response.json();
        if (!closed && text?.trim()) setInterimTranscript(text.trim());
      },
      onError: () => { if (!closed) { setMicEnabled(false); setError('Microphone interrupted. Enable it again to resume.'); } },
      onUtterance: async blob => {
        if (closed || voiceWorking.current) return;
        const bargeIn = speakingRef.current || audioBusyRef.current || busyRef.current || statusBusyRef.current;
        if (bargeIn) {
          stopAudio();
          abort.current?.abort();
          setBusy(false);
        }
        voiceWorking.current = true;
        setAudioBusy(true);
        setInterimTranscript('');
        try {
          const response = await checked(await fetch('/api/interface/transcribe', { method: 'POST', headers: { 'Content-Type': 'audio/wav' }, body: blob, signal: transcription.signal }));
          const { text } = await response.json();
          if (!closed && text?.trim()) {
            // Wait for the aborted interactive turn to release its server lock.
            // This is the dedicated barge-in handoff: no second turn is sent
            // while the previous stream is still unwinding.
            const deadline = performance.now() + 2500;
            while (!closed && (busyRef.current || statusBusyRef.current) && performance.now() < deadline) {
              await new Promise(resolve => setTimeout(resolve, 40));
            }
            if (!closed) await submitVoice.current(text.trim(), true);
          }
        } catch (e) {
          if (!closed) setError(e instanceof Error ? e.message : 'Transcription failed.');
        } finally {
          voiceWorking.current = false;
          if (!closed) setAudioBusy(false);
        }
      },
    }).then(async cleanup => {
      if (closed) await cleanup();
      else { stop = cleanup; setRecording(true); }
    }).catch(() => {
      if (!closed) { setMicEnabled(false); setError('Voice detection could not start. Check microphone permission and try again.'); }
    });
    return () => { closed = true; transcription.abort(); setRecording(false); setAudioBusy(false); setInterimTranscript(''); void stop?.(); };
  }, [voice, micEnabled, connected, status?.ready]);

  useEffect(() => {
    mounted.current = true;
    return () => { mounted.current = false; abort.current?.abort(); audioAbort.current?.abort(); playback.current?.pause(); if (playbackUrl.current) URL.revokeObjectURL(playbackUrl.current); };
  }, []);
  useEffect(() => {
    let stopped = false;
    let timeout: ReturnType<typeof setTimeout>;
    async function refreshSurfaces() {
      try {
        const [networkResponse, cameraResponse] = await Promise.all([
          fetch('/api/interface/network'),
          cameraMode ? fetch('/api/interface/camera/frame') : Promise.resolve(null),
        ]);
        if (!stopped && networkResponse.ok) setNetworkState(await networkResponse.json());
        if (!stopped && cameraResponse?.ok) { const frame = await cameraResponse.json(); setCameraFrame(frame.frame ? `data:image/jpeg;base64,${frame.frame}` : null); setCameraFaces(frame.faces || []); }
      } catch { /* connection indicator owns the visible error state */ }
      if (!stopped) timeout = setTimeout(refreshSurfaces, cameraMode ? 120 : 3000);
    }
    void refreshSurfaces();
    return () => { stopped = true; clearTimeout(timeout); };
  }, [cameraMode]);
  useEffect(() => { try { localStorage.setItem(storageKey, JSON.stringify(messages.slice(-200))); } catch { setError('This browser could not save the transcript.'); } }, [messages]);
  useEffect(() => { try { localStorage.setItem(telemetryVisibilityKey, String(showTelemetry)); } catch { /* optional UI preference */ } }, [showTelemetry]);
  useEffect(() => {
    let stopped = false;
    const controller = new AbortController();
    let timeout: ReturnType<typeof setTimeout>;
    async function refresh() {
      try { const response = await checked(await fetch('/api/interface/status', { signal: AbortSignal.any([controller.signal, AbortSignal.timeout(5000)]) })); const data: Status = await response.json(); if (!stopped) { setStatus(data); setResponseLanguage(data.response_language || 'auto'); setWebSearchMode(data.web_search_mode || 'off'); setResponseVerbosity(data.response_verbosity || 'concise'); setConnected(true); const fresh = (data.presence_messages || []).filter(item => !seenPresence.current.has(item.id)); fresh.forEach(item => seenPresence.current.add(item.id)); if (fresh.length) window.dispatchEvent(new CustomEvent('lumina-presence', { detail: fresh[fresh.length - 1] })); } }
      catch { if (!stopped) setConnected(false); }
      if (!stopped) timeout = setTimeout(refresh, document.hidden ? 10000 : 2000);
    }
    void refresh();
    return () => { stopped = true; clearTimeout(timeout); controller.abort(); };
  }, []);
  useEffect(() => { if (atBottom) transcript.current?.scrollTo({ top: transcript.current.scrollHeight }); }, [messages, atBottom, voice]);
  useEffect(() => {
    let stopped = false;
    void fetch('/api/interface/settings').then(checked).then(response => response.json()).then(result => {
      const configured = typeof result.values?.PERSONA_NAME === 'string' ? result.values.PERSONA_NAME.trim() : '';
      if (!stopped && configured) setPersonaName(configured);
    }).catch(() => {});
    return () => { stopped = true; };
  }, []);
  useEffect(() => { if (inspector) inspectorClose.current?.focus(); }, [inspector]);
  useEffect(() => { const handler = (e: KeyboardEvent) => { if (e.key === 'Escape') { closeInspector(); setSettings(false); setHistory(false); } }; window.addEventListener('keydown', handler); return () => window.removeEventListener('keydown', handler); }, []);
  useEffect(() => {
    const handler = (event: MouseEvent) => {
      const target = event.target as HTMLElement | null;
      const link = target?.closest('a[href="http://127.0.0.1:8080/settings"]');
      if (link) {
        event.preventDefault();
        openSettingsWindow('/settings');
      }
    };
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, []);

  function closeInspector() { setInspector(false); inspectorTrigger.current?.focus(); }
  function openSettingsWindow(path: string) {
    const settingsMatch = path.match(/^\/settings(?:\?tab=([^&]+))?/);
    const migrated = path === '/cognitive-dashboard' ? '/next/?view=health' : path === '/vision' ? '/next/?view=settings&tab=vision' : path === '/rss-feeds' ? '/next/?view=rss' : '';
    const target = migrated || (settingsMatch ? `/next/?view=settings${settingsMatch[1] ? `&tab=${encodeURIComponent(settingsMatch[1])}` : ''}` : path);
    window.open(target, `lumina-${path.replace(/[^a-z0-9]/gi, '-')}`, 'popup,width=1100,height=820,resizable=yes,scrollbars=yes');
  }
  async function changeLanguage(language: string) {
    const previous = responseLanguage;
    setResponseLanguage(language);
    try { await checked(await fetch('/api/interface/language', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ language }) })); }
    catch (e) { setResponseLanguage(previous); setError(e instanceof Error ? e.message : 'Language change failed.'); }
  }
  async function updatePreference(kind: 'web_search_mode' | 'response_verbosity', value: 'off' | 'auto' | 'always' | 'concise' | 'verbose') {
    const previous = kind === 'web_search_mode' ? webSearchMode : responseVerbosity;
    if (kind === 'web_search_mode') setWebSearchMode(value as 'off' | 'auto' | 'always'); else setResponseVerbosity(value as 'concise' | 'verbose');
    try { await checked(await fetch('/api/interface/preferences', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ [kind]: value }) })); }
    catch (e) { if (kind === 'web_search_mode') setWebSearchMode(previous as 'off' | 'auto' | 'always'); else setResponseVerbosity(previous as 'concise' | 'verbose'); setError(e instanceof Error ? e.message : 'Preference change failed.'); }
  }
  function cycleWebSearch() { const next = ({ off: 'auto', auto: 'always', always: 'off' } as const)[webSearchMode]; void updatePreference('web_search_mode', next); }
  function startNewConversation() { setMessages([]); setDraft(''); setError(''); }
  function stopAudio() {
    echoUntil.current = performance.now() + 400;
    operation.current++;
    speechGeneration.current++;
    speechQueue.current = [];
    audioAbort.current?.abort();
    playback.current?.pause(); playback.current = null;
    if (playbackUrl.current) URL.revokeObjectURL(playbackUrl.current);
    playbackUrl.current = null;
    setSpeaking(false); setAudioBusy(false); setLevel(0);
  }
  function stopTurn() { abort.current?.abort(); stopAudio(); }
  function leaveVoiceMode() {
    // The listener effect cleans up asynchronously; clear the UI gates now
    // so the normal composer is usable immediately after hanging up.
    voiceWorking.current = false;
    abort.current?.abort();
    setBusy(false);
    setVoice(false);
    setRecording(false);
    setAudioBusy(false);
    stopAudio();
  }
  async function toggleCamera() {
    try {
      const response = await checked(await fetch('/api/interface/camera/toggle', { method: 'POST' }));
      const result = await response.json();
      setCameraMode(Boolean(result.active));
      if (!result.active) { setCameraFrame(null); setCameraFaces([]); }
    } catch (e) { setError(e instanceof Error ? e.message : 'Camera unavailable.'); }
  }
  async function toggleNetwork() {
    setNetworkBusy(true);
    try { const response = await checked(await fetch('/api/interface/network/toggle', { method: 'POST' })); setNetworkState(await response.json()); setNetworkOpen(true); }
    catch (e) { setError(e instanceof Error ? e.message : 'Flux network action failed.'); }
    finally { setNetworkBusy(false); }
  }
  async function networkAction(action: 'ping' | 'ask' | 'dialogue', childId: string) {
    const child = networkState.children.find(item => item.id === childId);
    if (!child) return;
    const dialogueWindow = action === 'dialogue' ? window.open('about:blank', 'lumina-flux-dialogue', 'popup,width=980,height=760,resizable=yes,scrollbars=yes') : null;
    setNetworkBusy(true); setNetworkResult('');
    try {
      const payload = action === 'ask' ? { child_id: childId, text: draft.trim() } : action === 'dialogue' ? { child_id: childId, topic: draft.trim() || 'A shared reflection' } : { child_id: childId };
      const response = await checked(await fetch('/api/interface/network/action', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) }));
      const result = await response.json();
      if (action === 'ask') setDraft('');
      if (action === 'dialogue' && result.dialogue_id) {
        const url = `/next/?view=flux-dialogue&dialogue_id=${encodeURIComponent(result.dialogue_id)}`;
        if (dialogueWindow && !dialogueWindow.closed) dialogueWindow.location.href = url;
        else window.open(url, 'lumina-flux-dialogue', 'popup,width=980,height=760,resizable=yes,scrollbars=yes');
      }
      setNetworkResult(action === 'ping' ? `${child.name}: ${result.online ? 'online' : 'offline'}` : action === 'dialogue' ? `Dialogue window opened with ${child.name}` : result.response || 'Sent.');
      const refreshed = await fetch('/api/interface/network'); if (refreshed.ok) setNetworkState(await refreshed.json());
    } catch (e) { setNetworkResult(e instanceof Error ? e.message : 'Flux action failed.'); }
    finally { setNetworkBusy(false); }
  }

  async function speak(text: string) {
    stopAudio();
    const current = operation.current;
    const controller = new AbortController(); audioAbort.current = controller;
    setAudioBusy(true);
    try {
      const response = await checked(await fetch('/api/interface/speak', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text: text.slice(0, 8000) }), signal: controller.signal }));
      const blob = await response.blob();
      if (current !== operation.current) return;
      const url = URL.createObjectURL(blob); playbackUrl.current = url;
      const audio = new Audio(url); playback.current = audio;
      audio.onended = () => stopAudio(); audio.onerror = () => { stopAudio(); setError('Audio playback failed.'); };
      await audio.play(); setSpeaking(true);
    } catch (e) { if (!controller.signal.aborted) setError(e instanceof Error ? e.message : 'Speech failed.'); }
    finally { if (current === operation.current) setAudioBusy(false); }
  }

  function sentenceBoundary(text: string): number {
    const matches = [...text.matchAll(/[.!?](?:["'»)]*)?(?:\s+|$)/g)];
    const last = matches.at(-1);
    return last?.index === undefined ? -1 : last.index + last[0].length;
  }

  async function drainSpeechQueue() {
    if (speechDraining.current) return;
    speechDraining.current = true;
    const generation = speechGeneration.current;
    try {
      while (speechQueue.current.length && generation === speechGeneration.current) {
        const sentence = speechQueue.current.shift();
        if (!sentence?.trim()) continue;
        setAudioBusy(true);
        const controller = new AbortController();
        audioAbort.current = controller;
        try {
          const response = await checked(await fetch('/api/interface/speak', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: sentence.slice(0, 8000) }), signal: controller.signal,
          }));
          const blob = await response.blob();
          if (generation !== speechGeneration.current) break;
          const url = URL.createObjectURL(blob);
          const audio = new Audio(url);
          playback.current = audio;
          await new Promise<void>((resolve, reject) => {
            audio.onended = () => resolve();
            // stopAudio() pauses the current element when the user leaves
            // voice mode or starts a new turn. Resolve the queue immediately
            // so the audio state cannot remain busy after that cancellation.
            audio.onpause = () => resolve();
            audio.onerror = () => reject(new Error('Audio playback failed.'));
            void audio.play().then(() => { setSpeaking(true); }).catch(reject);
          });
          URL.revokeObjectURL(url);
          playback.current = null;
          setSpeaking(false);
        } catch (e) {
          if (!controller.signal.aborted && generation === speechGeneration.current) setError(e instanceof Error ? e.message : 'Speech failed.');
          break;
        } finally {
          if (audioAbort.current === controller) audioAbort.current = null;
        }
      }
    } finally {
      speechDraining.current = false;
      if (!speechQueue.current.length && generation === speechGeneration.current) setAudioBusy(false);
      if (generation === speechGeneration.current) setSpeaking(false);
    }
  }

  function enqueueSpeech(text: string) {
    const clean = visibleText(text).trim();
    if (!clean) return;
    speechQueue.current.push(clean);
    void drainSpeechQueue();
  }

  async function giveFeedback(message: Message, positive: boolean) {
    if (message.feedback || !message.text.trim()) return;
    const feedback = positive ? 'positive' : 'negative';
    setMessages(current => current.map(item => item.id === message.id ? { ...item, feedback } : item));
    try {
      await checked(await fetch('/api/interface/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ positive, response: message.text, message_id: message.id }),
      }));
    } catch (e) {
      setMessages(current => current.map(item => item.id === message.id ? { ...item, feedback: undefined } : item));
      setError(e instanceof Error ? e.message : 'Feedback could not be recorded.');
    }
  }

  async function toggleTelemetry() {
    const next = !showTelemetry;
    setShowTelemetry(next);
    if (!next) return;
    try {
      const response = await checked(await fetch('/api/interface/telemetry'));
      const telemetry = await response.json() as CognitiveTelemetry;
      setMessages(current => {
        let index = current.length - 1;
        while (index >= 0 && current[index].role !== 'assistant') index -= 1;
        if (index < 0) return current;
        const updated = current.slice();
        updated[index] = { ...updated[index], telemetry };
        return updated;
      });
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Cognitive telemetry could not be loaded.');
    }
  }

  async function send(text = draft, automatic = false) {
    text = text.trim();
    if (!text || busy || (!automatic && audioBusy) || !connected || !status?.ready || status.busy) return;
    stopAudio(); setError(''); setDraft(''); setBusy(true); setAtBottom(true);
    const assistantId = crypto.randomUUID();
    setMessages(previous => [...previous, { id: crypto.randomUUID(), role: 'user', text }, { id: assistantId, role: 'assistant', text: '' }]);
    const controller = new AbortController(); abort.current = controller;
    let accumulated = ''; let reasoningAccumulated = ''; let complete = false; let spokenCursor = 0; let spokenSentenceCount = 0;
    try {
      const response = await checked(await fetch('/api/interface/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }), signal: controller.signal }));
      if (!response.body) throw new Error('The response stream is unavailable.');
      for await (const event of readEvents(response.body)) {
        if (event.type === 'error') throw new Error(event.message);
        if (event.type === 'reasoning') {
          // Reasoning is a separate transient stream. Do not clear visible
          // answer text when a provider emits reasoning after a final delta.
          reasoningAccumulated = appendReasoning(reasoningAccumulated, String(event.text || ''));
          if (reasoningAccumulated) {
            window.dispatchEvent(new CustomEvent('lumina-reasoning', { detail: reasoningAccumulated }));
          }
        }
        if (event.type === 'delta') {
          accumulated += event.text;
          setMessages(previous => previous.map(m => m.id === assistantId ? { ...m, text: visibleText(accumulated) } : m));
          if (voiceActive.current) {
            const visible = visibleText(accumulated);
            // The first paragraph is the immediate answer. The remaining
            // paragraphs stay in the transcript but are not read aloud.
            const shortAnswer = visible.split(/\n\s*\n/)[0];
            const available = shortAnswer.slice(spokenCursor);
            const boundary = sentenceBoundary(available);
            if (boundary > 0 && spokenSentenceCount < 2) {
              enqueueSpeech(available.slice(0, boundary));
              spokenCursor += boundary;
              spokenSentenceCount += 1;
            }
          }
        }
        if (event.type === 'telemetry' && event.data) {
          setMessages(previous => previous.map(m => m.id === assistantId ? { ...m, telemetry: event.data as CognitiveTelemetry } : m));
        }
        if (event.type === 'web_sources' && Array.isArray(event.sources)) {
          setMessages(previous => previous.map(m => m.id === assistantId ? { ...m, webSources: event.sources as WebSource[] } : m));
        }
        if (event.type === 'done') complete = true;
      }
      if (!complete) throw new Error('Connection ended before the response completed.');
      if (voiceActive.current) {
        const shortAnswer = visibleText(accumulated).split(/\n\s*\n/)[0];
        const remaining = spokenSentenceCount < 2 ? shortAnswer.slice(spokenCursor).trim() : '';
        if (remaining) enqueueSpeech(remaining);
      }
    } catch (e) {
      setMessages(previous => previous.map(m => m.id === assistantId ? { ...m, stopped: true } : m));
      if (!controller.signal.aborted) { setError(e instanceof Error ? e.message : 'The request failed.'); setDraft(text); }
    } finally { setBusy(false); abort.current = null; composer.current?.focus(); }
  }

  async function finishRecording(transcribe = true) {
    if (voice) setMicEnabled(false);
  }

  async function toggleRecording() {
    setError('');
    if (voice) setMicEnabled(enabled => !enabled);
    else { setMicEnabled(true); setVoice(true); }
  }

  const ready = connected && status?.ready;
  const stateLabel = !connected ? 'Disconnected' : !status?.ready ? 'Starting' : speaking ? 'Speaking' : audioBusy ? 'Preparing audio' : busy || status.busy ? 'Processing' : voice && !micEnabled ? 'Microphone muted' : voice && interimTranscript ? 'Hearing you...' : recording ? 'Listening' : voice ? 'Starting microphone' : 'Present';
  const organismMode = recording ? 'listening' : audioBusy ? 'processing' : speaking || busy || Boolean(status?.busy) ? 'responding' : 'idle';
  const selectedEvent = status?.events.find(event => event.id === eventId);
  const revealOrchestration = () => document.querySelector('.cognition-activity-section')?.scrollIntoView({ behavior: 'smooth', block: 'start' });

  return <div className={`app ${voice ? 'voice-mode' : ''} ${inspector ? 'inspector-open' : ''}`} style={{ '--inspector-width': `${panelWidth}px` } as React.CSSProperties}>
    <header className="topbar"><div className="brand-symbol"><Sparkles size={23}/></div><a className="brand" href="/next/">{personaName}<span>PERSONAL COGNITIVE SPACE</span></a><div className={`connection ${connected ? 'online' : ''}`}><i/>{connected ? status?.ready ? 'Connected' : 'Starting' : 'Disconnected'}</div><div className="telemetry-strip"><span>Memory <b>{status?.providers ? 'FAISS' : 'Unavailable'}</b></span><span>STT <b>{status?.providers.stt ?? 'Unavailable'}</b></span><span>TTS <b>{status?.providers.tts ?? 'Unavailable'}</b></span><span className={cameraMode ? 'telemetry-live' : ''}><Eye size={12}/> {cameraMode ? 'Camera' : 'Vision off'}</span><span className={networkState.enabled ? 'telemetry-live' : ''}><Network size={12}/> {networkState.enabled ? `Flux ${networkState.children.length}` : 'Flux off'}</span></div><div className="header-actions"><label className="language-control" title="Response language"><span>Lang</span><select aria-label="Response language" value={responseLanguage} onChange={event => void changeLanguage(event.target.value)}>{languageOptions.map(([value, label]) => <option value={value} key={value}>{label}</option>)}</select></label><span className="session-label">Conversation</span><IconButton label={cameraMode ? 'Close camera mode' : 'Open camera mode'} active={cameraMode} onClick={() => void toggleCamera()}><Eye size={19}/></IconButton><IconButton label={networkState.enabled ? 'Open Flux network' : 'Enable Flux network'} active={networkOpen} disabled={networkBusy} onClick={() => networkState.enabled ? setNetworkOpen(open => !open) : void toggleNetwork()}><Network size={19}/></IconButton><button ref={inspectorTrigger} className={`icon-button ${inspector ? 'active' : ''}`} aria-label="Open cognitive inspector" aria-expanded={inspector} title="Cognitive inspector" onClick={() => setInspector(!inspector)}><PanelRightOpen size={19}/></button><IconButton label={voice ? 'Return to chat' : 'Enter voice mode'} active={voice} onClick={() => voice ? leaveVoiceMode() : setVoice(true)}><AudioLines size={20}/></IconButton></div></header>
    <nav className="rail" aria-label="Main navigation"><IconButton label="Conversation" active={!voice} onClick={() => { if (voice) leaveVoiceMode(); }}><MessageCircle size={21}/></IconButton><IconButton label="Conversation history" active={history} onClick={() => setHistory(!history)}><History size={21}/></IconButton><IconButton label="Cognitive activity" active={inspector} onClick={() => { setTab('Activity'); setInspector(true); }}><Activity size={21}/></IconButton><div className="rail-spacer"/><IconButton label="Interface settings" active={settings} onClick={() => openSettingsWindow('/settings')}><Settings size={20}/></IconButton></nav>
    <main className="workspace">
      <section className="presence" aria-label="Lumina presence"><div className="presence-heading"><span className="eyebrow">LUMINA / LIVE PRESENCE</span><IconButton label={voice ? 'Return to chat' : 'Expand voice view'} onClick={() => voice ? leaveVoiceMode() : setVoice(true)}><Expand size={15}/></IconButton></div>{cameraMode && <div className="camera-orb" aria-label="Live camera feed"><div className="camera-orb-label"><Eye size={13}/>Camera live</div>{cameraFrame ? <img src={cameraFrame} alt="Live camera view"/> : <div className="camera-empty">Waiting for camera</div>}{cameraFaces.map((face, index) => { const loc = face.location || {}; const left = ((loc.left || 0) / 640) * 100; const top = ((loc.top || 0) / 480) * 100; const width = (((loc.right || 0) - (loc.left || 0)) / 640) * 100; const height = (((loc.bottom || 0) - (loc.top || 0)) / 480) * 100; return <div className="camera-face-box" key={`${face.name}-${index}`} style={{ left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%` }}><span>{face.name}</span></div>; })}</div>}<div className="organism-stage" onMouseEnter={() => setOrganismHover(true)} onMouseLeave={() => setOrganismHover(false)} onFocusCapture={() => setOrganismHover(true)} onBlurCapture={(event) => { if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setOrganismHover(false); }}><div className={`organism-quickbar ${organismHover ? 'is-visible' : ''}`} aria-label="Quick settings"><IconButton label="Open cognitive dashboard" onClick={() => openSettingsWindow('/cognitive-dashboard')}><Activity size={17}/></IconButton><IconButton label="Open memory settings" onClick={() => openSettingsWindow('/settings?tab=memory')}><Database size={17}/></IconButton><IconButton label="Open voice settings" onClick={() => openSettingsWindow('/settings?tab=voice')}><AudioLines size={17}/></IconButton><IconButton label="Open camera settings" onClick={() => openSettingsWindow('/vision')}><Eye size={17}/></IconButton><IconButton label="Open Flux network settings" onClick={() => openSettingsWindow('/orchestrator')}><Network size={17}/></IconButton><IconButton label="Open interface settings" onClick={() => openSettingsWindow('/settings')}><Settings size={17}/></IconButton></div><Organism active={busy || Boolean(status?.busy)} energy={level} mode={organismMode} reduced={reduced || !connected}/></div><div className="presence-caption"><span className={`state-dot ${recording || speaking ? 'bright' : ''}`}/><span>{stateLabel}</span><span className="caption-divider"/><span className="presence-emotion">{status?.emotional_state || (connected ? 'Present and attentive' : 'Awaiting connection')}</span>{status?.life_stage && <><span className="caption-divider"/><span className="presence-stage">{status.life_stage}{status.current_age != null ? ` · age ${status.current_age.toFixed(1)}` : ''}</span></>}<span className="caption-divider"/>{connected ? 'Here with you' : 'Awaiting connection'}</div><div className="presence-foot"><span>01 / {voice ? 'VOICE' : cameraMode ? 'CAMERA' : 'CONVERSATION'}</span><span>{personaName.toUpperCase()}</span></div></section>
      <section className="conversation" aria-label="Conversation"><div className="conversation-heading"><div><span className="eyebrow">YOUR SPACE TO THINK</span><h1>{voice ? 'In conversation.' : 'A conversation, unfolding.'}</h1></div><div className="conversation-tools"><IconButton label="Reveal orchestration field" onClick={revealOrchestration}><Workflow size={16}/></IconButton><IconButton label="Open cognitive dashboard" onClick={() => openSettingsWindow('/cognitive-dashboard')}><Activity size={16}/></IconButton><IconButton label="Open RSS feed management" onClick={() => openSettingsWindow('/rss-feeds')}><Rss size={16}/></IconButton><IconButton label="Interface settings" onClick={() => setSettings(true)}><Settings size={16}/></IconButton><IconButton label="New conversation" onClick={startNewConversation}><MessageCircle size={16}/></IconButton><IconButton label={`Web search: ${webSearchMode}`} active={webSearchMode !== 'off'} onClick={cycleWebSearch}><Search size={16}/></IconButton><IconButton label={responseVerbosity === 'verbose' ? 'Extended responses on' : 'Concise responses on'} active={responseVerbosity === 'verbose'} onClick={() => void updatePreference('response_verbosity', responseVerbosity === 'verbose' ? 'concise' : 'verbose')}><ChevronRight size={16} style={{ transform: responseVerbosity === 'verbose' ? 'rotate(90deg)' : 'rotate(0deg)' }}/></IconButton><span className="day-label">{new Date().toLocaleDateString(undefined, { month: 'short', day: 'numeric' })}</span></div></div>
        <div className="transcript" ref={transcript} onScroll={e => { const el = e.currentTarget; setAtBottom(el.scrollHeight - el.scrollTop - el.clientHeight < 90); }}>
          {messages.length === 0 ? <div className="empty-conversation"><Sparkles size={24}/><h2>Where shall we begin?</h2><p>{ready ? 'I am here.' : connected ? `${personaName} is waking up.` : `Waiting for ${personaName} to connect.`}</p></div> : messages.map(m => <article key={m.id} id={`message-${m.id}`} className={`message ${m.role}`}><div className="message-author">{m.role === 'assistant' ? <Sparkles size={16}/> : <span className="user-mark">F</span>}<span>{m.role === 'assistant' ? personaName : 'You'}</span></div><div className="message-content"><Markdown>{m.text || (m.stopped ? 'Response stopped.' : 'Thinking...')}</Markdown>{m.stopped && m.text && <small className="muted">Response stopped</small>}</div>{m.role === 'assistant' && m.webSources?.length ? <div className="web-provenance"><span>Sources used</span>{m.webSources.map((source, index) => <a key={source.url + index} href={source.url} target="_blank" rel="noreferrer"><b>{source.kind === 'page' ? 'Page read' : source.kind === 'fetch_failed' ? 'Fetch failed' : 'Search result'}</b> {source.title || source.url}</a>)}</div> : null}{m.role === 'assistant' && showTelemetry && m.telemetry && <TelemetryTrace data={m.telemetry}/>} {m.role === 'assistant' && m.text && <div className="message-actions"><IconButton label="Copy response" onClick={async () => { try { await navigator.clipboard.writeText(m.text); setCopied(m.id); } catch { setError('Clipboard is unavailable.'); } }}>{copied === m.id ? <Check size={14}/> : <Copy size={14}/>}</IconButton><IconButton label="Read response aloud" disabled={!ready || audioBusy || recording} onClick={() => void speak(m.text)}><Volume2 size={15}/></IconButton><span className="feedback-label">Helpful?</span><IconButton label="Mark response helpful" active={m.feedback === 'positive'} aria-pressed={m.feedback === 'positive'} disabled={Boolean(m.feedback)} onClick={() => void giveFeedback(m, true)}><ThumbsUp size={14}/></IconButton><IconButton label="Mark response unhelpful" active={m.feedback === 'negative'} aria-pressed={m.feedback === 'negative'} disabled={Boolean(m.feedback)} onClick={() => void giveFeedback(m, false)}><ThumbsDown size={14}/></IconButton></div>}</article>)}
        </div>{networkOpen && <section className="flux-panel" aria-label="Flux network"><div className="flux-heading"><div><span className="eyebrow">LUMINA NETWORK / FLUX</span><h2>Network dialogue</h2></div><div className="flux-heading-actions"><span className={`network-indicator ${networkState.enabled ? 'online' : ''}`}><i/>{networkState.enabled ? 'Enabled' : 'Disabled'}</span><IconButton label="Refresh Flux network" disabled={networkBusy} onClick={() => { void fetch('/api/interface/network').then(response => response.json()).then(setNetworkState); }}><RefreshCw size={15}/></IconButton></div></div>{networkState.children.length === 0 ? <p className="muted">No Flux instances configured.</p> : <div className="flux-children">{networkState.children.map(child => <div className="flux-child" key={child.id}><div className="flux-child-title"><span className={`state-dot ${child.online ? 'bright' : ''}`}/><strong>{child.name}</strong><span className="flux-role">{child.role}</span><span className="flux-latency">{child.online ? `${Math.round(child.latency_ms)} ms` : 'offline'}</span></div><div className="flux-child-url">{child.url}</div><div className="flux-actions"><button onClick={() => void networkAction('ping', child.id)} disabled={networkBusy}><Wifi size={13}/>Ping</button><button onClick={() => void networkAction('ask', child.id)} disabled={networkBusy || !draft.trim()}><Send size={13}/>Ask</button><button onClick={() => void networkAction('dialogue', child.id)} disabled={networkBusy}><Play size={13}/>Dialogue</button></div></div>)}</div>}{networkResult && <div className="flux-result" role="status">{networkResult}</div>}<button className="flux-disable" onClick={() => void toggleNetwork()} disabled={networkBusy}>{networkState.enabled ? 'Disable Flux network' : 'Enable Flux network'}</button></section>}
        {!atBottom && <button className="jump-button" onClick={() => setAtBottom(true)}><ArrowDown size={15}/>Latest</button>}
        {voiceMemSpace?.status?.enabled && <VoiceMemSpace data={voiceMemSpace}/>}
        <div className="compose-area">{error && <div className="error" role="alert"><span>{error}</span><IconButton label="Dismiss error" onClick={() => setError('')}><X size={14}/></IconButton></div>}
          {voice && <div className="voice-status"><span className="voice-state"><AudioLines size={20}/>{stateLabel}</span><div className="call-controls"><IconButton label={micEnabled ? 'Mute microphone' : 'Resume microphone'} className={`call-microphone ${!micEnabled ? 'is-recording' : ''}`} active={micEnabled} onClick={() => void toggleRecording()}>{micEnabled ? <Mic size={22}/> : <MicOff size={22}/>}</IconButton><IconButton label="End voice mode" className="hangup" onClick={() => { void finishRecording(false); stopTurn(); setVoice(false); }}><Phone size={22} style={{ transform: 'rotate(135deg)' }}/></IconButton></div></div>}
          <form className={`composer ${recording ? 'recording' : ''}`} onSubmit={e => { e.preventDefault(); void send(); }}><textarea ref={composer} value={draft} maxLength={8000} rows={2} placeholder={recording ? 'Listening...' : 'Message Lumina...'} aria-label="Message Lumina" onChange={e => setDraft(e.target.value)} onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.nativeEvent.isComposing) { e.preventDefault(); void send(); } }}/><div className="composer-bottom"><span className="compose-status"><i/>{voice ? micEnabled ? 'Always listening' : 'Microphone muted' : audioBusy ? 'Audio processing' : 'Conversation'}</span><div className="composer-actions"><IconButton label={showTelemetry ? 'Hide cognitive telemetry' : 'Show cognitive telemetry'} type="button" active={showTelemetry} aria-pressed={showTelemetry} onClick={() => void toggleTelemetry()}><Activity size={18}/></IconButton>{!voice && (speaking || audioBusy ? <IconButton label="Stop audio" type="button" onClick={stopAudio}><Pause size={18}/></IconButton> : <IconButton label="Record voice message" type="button" disabled={!ready || busy} onClick={() => void toggleRecording()}><Mic size={19}/></IconButton>)}{busy ? <IconButton label="Stop response" type="button" className="send" onClick={stopTurn}><Square size={16}/></IconButton> : <IconButton label="Send message" type="submit" className="send" disabled={!draft.trim() || !ready || Boolean(status?.busy) || recording || audioBusy}><ArrowUp size={19}/></IconButton>}</div></div></form><div className="composer-foot"><span>{voice ? micEnabled ? 'Microphone active' : 'Microphone muted' : showTelemetry ? 'Cognitive telemetry visible' : 'Private workspace'}</span><span>{busy ? 'Responding' : 'Lumina'}</span></div>
        </div>
      </section>
    </main>
    <section className="cognition-activity-section"><CognitiveActivity data={healthSnapshot} mode={organismMode} connected={connected} reduced={reduced} personaName={personaName}/></section>
    <section className="cognitive-health" aria-label="Cognitive Health Dashboard"><CognitiveHealth onData={setHealthSnapshot}/></section>
    {inspector && <aside className="inspector" aria-label="Cognitive inspector"><div className="resize-handle" role="separator" aria-label="Inspector width" aria-orientation="vertical" aria-valuenow={panelWidth} aria-valuemin={340} aria-valuemax={520} tabIndex={0} onKeyDown={e => { if (e.key === 'ArrowLeft' || e.key === 'ArrowRight') { e.preventDefault(); setPanelWidth(w => Math.max(340, Math.min(520, w + (e.key === 'ArrowLeft' ? 20 : -20)))); } }} onPointerDown={e => e.currentTarget.setPointerCapture(e.pointerId)} onPointerMove={e => { if (e.currentTarget.hasPointerCapture(e.pointerId)) setPanelWidth(Math.max(340, Math.min(520, window.innerWidth - e.clientX))); }} onPointerUp={e => e.currentTarget.releasePointerCapture(e.pointerId)}/><div className="inspector-heading"><div><span className="eyebrow">LIVE INSPECTOR</span><h2>Cognition</h2></div><button ref={inspectorClose} className="icon-button" aria-label="Close cognitive inspector" onClick={closeInspector}><X size={19}/></button></div><div className="tabs" role="tablist">{tabs.map(t => <button key={t} role="tab" aria-selected={tab === t} onClick={() => { setTab(t); setEventId(null); }}>{t}</button>)}</div><div className="inspector-body" role="tabpanel">
      {!connected && <p className="stale">Disconnected. {status ? 'Showing last received state.' : 'Telemetry unavailable.'}</p>}
      {tab === 'Overview' && <><div className="section-label">CURRENT STATE</div><div className="large-state"><span className="state-dot"/>{stateLabel}</div><dl className="metrics"><div><dt>Runtime</dt><dd>{status?.ready ? 'Ready' : 'Unavailable'}</dd></div><div><dt>Cognitive cycles</dt><dd>{status?.cycle ?? 'Unavailable'}</dd></div><div><dt>Uptime</dt><dd>{status ? `${Math.floor(status.uptime / 60)} min` : 'Unavailable'}</dd></div></dl><div className="section-label">RUNTIME MODULES</div>{['persona', 'organism', 'loop'].map(name => <div className="module-row" key={name}><span>{name === 'loop' ? 'Cognitive loop' : name.charAt(0).toUpperCase() + name.slice(1)}</span><span className={status?.engines[name] ? 'available' : 'muted'}>{status?.engines[name] ? 'Available' : 'Unavailable'}</span></div>)}<div className="section-label">RECENT ACTIVITY</div>{renderEvents(4)}</>}
      {tab === 'Activity' && <><div className="section-label">CONVERSATION EVENTS</div>{renderEvents(40)}{selectedEvent && <div className="event-detail"><button className="text-button" onClick={() => setEventId(null)}><ArrowLeft size={14}/>Back</button><h3>{selectedEvent.label}</h3><dl><dt>Recorded</dt><dd>{new Date(selectedEvent.timestamp * 1000).toLocaleString()}</dd><dt>Source</dt><dd>Interface conversation adapter</dd><dt>Turn ID</dt><dd>{selectedEvent.turn_id ?? 'Unavailable'}</dd></dl></div>}</>}
      {tab === 'Diagnostics' && <><div className="section-label">LAST MEASURED LATENCY</div><dl className="metrics">{[['first_token_ms', 'First token'], ['turn_ms', 'Full response'], ['stt_ms', 'Transcription'], ['tts_ms', 'Speech synthesis']].map(([key, label]) => <div key={key}><dt>{label}</dt><dd>{status?.timings[key] !== undefined ? `${status.timings[key]} ms` : 'Unavailable'}</dd></div>)}</dl><div className="section-label">AUDIO PROVIDERS</div><dl className="metrics"><div><dt>Speech recognition</dt><dd>{status?.providers.stt ?? 'Unavailable'}</dd></div><div><dt>Speech synthesis</dt><dd>{status?.providers.tts ?? 'Unavailable'}</dd></div></dl><a className="text-button" href="http://127.0.0.1:8080/settings" target="_blank" rel="noreferrer">Provider settings<ChevronRight size={14}/></a></>}
    </div><div className="inspector-foot"><i className={connected ? 'live-dot' : ''}/>{status ? `Updated ${new Date(status.timestamp * 1000).toLocaleTimeString()}` : 'Awaiting telemetry'}</div></aside>}
    {history && <div className="history-panel"><div className="panel-title"><h2>History</h2><IconButton label="Close history" onClick={() => setHistory(false)}><X size={18}/></IconButton></div>{messages.filter(m => m.role === 'user').length === 0 ? <p className="muted">No messages yet.</p> : messages.filter(m => m.role === 'user').map(m => <button className="history-item" key={m.id} onClick={() => { setHistory(false); setVoice(false); setAtBottom(false); setTimeout(() => document.getElementById(`message-${m.id}`)?.scrollIntoView({ block: 'center' }), 0); }}><MessageCircle size={15}/><span>{m.text}</span><ChevronRight size={14}/></button>)}</div>}
    {settings && <dialog open className="settings-dialog" aria-label="Interface settings"><div className="panel-title"><h2>Interface</h2><IconButton label="Close settings" onClick={() => setSettings(false)}><X size={18}/></IconButton></div><label className="setting-row">Reduce motion<input type="checkbox" checked={reduced} onChange={e => setReduced(e.target.checked)}/></label><a className="text-button" href="http://127.0.0.1:8080/settings" target="_blank" rel="noreferrer">Lumina settings<ChevronRight size={15}/></a></dialog>}
  </div>;

  function renderEvents(limit: number) {
    const events = status?.events.slice(-limit).reverse() || [];
    return events.length ? <div className="event-list">{events.map(event => <button key={event.id} onClick={() => { setTab('Activity'); setEventId(event.id); }}><i/><div><span>{event.label}</span><time>{new Date(event.timestamp * 1000).toLocaleTimeString()}</time></div><ChevronRight size={14}/></button>)}</div> : <p className="muted no-events">No recorded events.</p>;
  }
}
