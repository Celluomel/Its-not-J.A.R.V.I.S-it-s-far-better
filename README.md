# Cognitive Organism - Beyond J.A.R.V.I.S.

## A research-grade cognitive harness

This project is a **research-grade cognitive harness for building, instrumenting and testing persistent artificial cognition**. It is designed for researchers, engineers and technically curious testers who want to examine what happens when a language model is placed inside a longer-lived system with memory, perception, drives, goals, planning, self-correction and measurable internal state.

The language model is a replaceable reasoning component. The harness supplies the surrounding machinery: stateful cognitive streams, a global workspace, background processing, world-model grounding, action selection, learning signals and inspection surfaces. That distinction is the product.

This repository contains an experimental **cognitive organism**: a local-first research system that combines memory, perception, voice, reflection, planning, self-correction and an evolving visual representation of internal activity.

The ambition is to investigate how persistent state, internal drives, world models, memory, perception and self-evaluation can be assembled into a continuously operating cognitive architecture. It is not a chatbot project and is not presented as artificial general intelligence.

Its value will come from careful engineering, explicit instrumentation, honest evaluation and repeated testing with real people.

## Position in the current landscape

Most public AI products optimise for a polished response loop. Most agent frameworks optimise for tool execution. Most companion systems optimise for continuity of tone and remembered preferences. Cognitive research prototypes often demonstrate one mechanism in isolation. This project occupies a different position: it is an **open, local-first integration and evaluation harness** for studying how these mechanisms interact over time.

| Capability | Consumer assistants | Chatbot / companion products | Agent frameworks | Cognitive research prototypes | This harness |
|---|---|---|---|---|---|
| Persistent episodic and semantic state | Usually limited or opaque | Often preference-focused | Usually application-defined | Often narrow or experimental | Built in and inspectable |
| User and interlocutor modelling | Account/profile layer | Strong persona continuity | Usually left to the developer | Varies by experiment | Persistent profiles linked to interaction and perception |
| Perception beyond text | Product-specific | Usually limited | Connector-dependent | Often modality-specific | Voice, camera, vision memory and face recognition paths |
| Autonomous background cognition | Mostly hidden or task-triggered | Proactive messaging, usually opaque | Schedulers and workers | Frequently simulated or narrow | Goals, thought streams, workspace competition and orchestration |
| Intention, causal and temporal reasoning | Prompt-dependent | Prompt-dependent | Tool or workflow dependent | Research-specific | Dedicated reasoning and planning subsystems with state |
| Self-correction from outcomes | Mostly implicit | Feedback may be product-level | Developer-defined | Often evaluated experimentally | Corrections, evidence, competing strategies and learned behavioural rules |
| Inspectability and telemetry | Low | Low to medium | Logs and traces | Instrumentation varies | Cognitive traces, health data, subsystem outputs and 3D state view |
| Local model and hardware control | Rarely central | Rarely central | Possible | Common | First-class LM Studio, Ollama, local voice and camera operation |

This is a capability comparison, not a claim that the organism outperforms every commercial system on general language quality. The comparison concerns **architecture, persistence and inspectability**. Response quality still depends on the selected model, prompt context, hardware and the quality of the evidence collected by each subsystem.

## Comparison with public cognitive research

This project is best understood alongside public cognitive architectures and agent-memory research, not only alongside commercial assistants:

| Public project or research line | Primary contribution | Difference in emphasis here |
|---|---|---|
| [Soar](https://github.com/SoarGroup/Soar) | A long-running general cognitive architecture with production rules, working memory and decision cycles | This harness uses a Python-first, LLM-assisted stack and focuses on multimodal interaction, persistent personal context and live operational telemetry |
| [OpenCog / Hyperon](https://github.com/opencog) | Symbolic and neuro-symbolic knowledge representation, reasoning and cognitive architecture research | This harness is less symbolic at its core and instead studies how memory, embeddings, model calls, drives and world-state signals can be composed into a continuously running organism |
| [ACT-R](https://act-r.psy.cmu.edu/) | A psychologically grounded architecture for modelling human cognition and task performance | This project is an engineering testbed, not a validated computational theory of human cognition; its value is in instrumented integration and reproducible failure analysis |
| [Generative Agents](https://github.com/joonspk-research/generative_agents) | Memory, reflection and planning for believable agents in a simulated social world | This harness carries those ideas into a persistent local process with real user interaction, voice, camera perception, correction learning and subsystem-level traces |
| [MemGPT / Letta](https://github.com/letta-ai/letta) | Stateful agents with explicit memory management and context control | This harness treats memory as one part of a broader cognitive ecology that also includes attention, affect, identity, goals, planning, perception and orchestration |
| ReAct-style and tool-agent frameworks | Structured reasoning followed by tool use and external action | This harness investigates background cognition and internal state evolution even when no tool call or user task is active |

The comparison is about **research posture and system boundaries**, not a leaderboard. These projects are important reference points, and this repository should be evaluated against them with shared tasks: memory retrieval precision, planning validity, temporal consistency, correction retention, interruption latency, grounding accuracy and long-horizon behavioural stability.

## Why “harness” matters

The harness is intended to make cognitive claims testable rather than merely conversational. A subsystem should be able to produce observable functional output:

- memory should retrieve relevant prior evidence, not just store text;
- perception should update a grounded world state and identify people when configured;
- intention reasoning should distinguish the main action from incidental details;
- planning should maintain dependencies, time constraints, progress and revision;
- self-correction should preserve lessons from explicit errors and test competing repairs;
- background cognition should feed the global workspace without hijacking active interaction;
- telemetry should expose enough provenance to explain why a subsystem acted.

The project therefore treats failures, empty outputs and “no evidence” states as research data. A high score is earned through repeatable tests, not through a convincing narrative generated by the model.

The detailed architecture and evaluation protocol are documented in [A Persistent Cognitive Organism: Architecture, Mechanisms and Evaluation Protocol](docs/COGNITIVE_ORGANISM_RESEARCH_PAPER.md).

## A useful research companion

The research harness is also intended to be useful in ordinary intellectual work. Its companion layer can use the configured web-research tools to:

- investigate scientific subjects and connect a discussion to papers, including **arXiv** sources;
- fetch and compare current information or news when recency matters;
- inspect a website supplied as a URL in the conversation;
- discuss a proposed goal, identify assumptions and constraints, and examine its feasibility;
- help refine an argument, question, hypothesis or design through back-and-forth dialogue;
- retain relevant user preferences, project context and prior corrections so later discussion has continuity.

This is not a separate chatbot persona. The useful distinction is that the language model supplies language and synthesis while the organism contributes memory, grounding, source retrieval, goal context and observable reasoning signals. Research tasks can therefore be examined both as conversation and as evidence flowing through the cognitive system.

Psychology-related conversations should be treated as reflective exploration and question refinement, not diagnosis, therapy or professional advice. Health, legal, safety-critical and other high-consequence decisions require qualified human professionals and appropriate primary sources.

## A collaborative organism, not a static demo

The organism is designed to evolve through a feedback loop between the person using it and the cognitive system being studied. A user can propose a new capability, tool, sensor or embodiment idea from the **Capability incubator**. The organism can inspect that proposal using its current goals, memories, Workspace state and subsystem context, while the user remains the external evaluator who decides whether the result is useful and trustworthy.

The development loop is:

```mermaid
flowchart LR
    U[User proposal] --> P[Persisted proposal]
    P --> A[Organism analysis]
    A --> E[Bounded local experiment]
    E --> M[Baseline and repeated measurements]
    M --> V[Provisional result]
    V --> H[Long-run confirmation]
    H --> C[User evaluation and next proposal]
    C --> U
```

This separates three claims that are often confused in AI demonstrations:

- **Proposal:** an idea for a possible capability, not a claim that it exists;
- **Experiment:** a bounded, asynchronous and reversible implementation or measurement;
- **Verified capability:** a result that survives repeated evidence and, where relevant, long-run stabilization.

For example, the Workspace experiment can establish a baseline, observe repeated changes in cognitive signal diversity and activity, then move through `provisionally_verified` and `stabilizing` before accepting a capability as durable. An inconclusive result remains useful research data. It tells the user and the organism which hypothesis needs refinement rather than turning a promising narrative into a permanent belief.

This is the intended relationship: the organism contributes persistent context, internal analysis and autonomous measurement; the user contributes direction, judgment, corrections and real-world evaluation. New testers are invited to propose capabilities, challenge the organism's conclusions, reproduce experiments and report where the internal evidence does not match the visible behaviour.

## System at a glance

```mermaid
flowchart LR
    P[Perception\ntext · voice · vision] --> W[Global Workspace]
    W --> A[Attention and\ninternal state]
    A --> R[Reasoning and\nworld model]
    R --> G[Goals and\nlong-horizon plans]
    G --> X[Actions and\nsubsystem outcomes]
    X --> C[Self-correction\nand cognitive learning]
    C --> M[(Persistent memory)]
    M --> W
    W --> O[Inspectable organism\nvisualisation and traces]
```

## Cognitive cycle

```mermaid
flowchart TB
    subgraph I[Interaction path]
        E[User or environment<br/>text · voice · camera] --> P[Cognitive organism<br/>input and context assembly]
        P --> R[Relevant memory<br/>identity · preferences · world state]
        R --> LLM[Configured local or remote LLM]
        LLM --> O[Response<br/>text · TTS · visible vision result]
        O --> F[Outcome and correction evidence]
    end

    subgraph B[Background path: InternalThoughtLoop]
        T[Slow cognitive cycle] --> D[Drives, affect, identity<br/>curiosity and pressure]
        D --> C[Workspace competition<br/>attention and dominant thought]
        C --> G[Goals, intention and planning]
        G --> X[Background actions<br/>reflection · experiments · maintenance]
    end

    P --> W[Global Workspace<br/>shared observable state]
    C --> W
    X --> W
    W --> R
    F --> M[Learning and persistence<br/>memory · skills · self-correction]
    M --> W
```

The two paths are deliberately distinct. A user interaction can publish signals to the Workspace and receive context from it, while the background loop continues on its own schedule. Background cognition does not automatically become a chat response: it must first compete for Workspace attention, pass through the relevant subsystem, and produce an observable outcome.

## What the organism does

- **Interaction layer:** local or remote OpenAI-compatible model providers are used as one component of the organism, not as the organism itself.
- **Memory:** episodic and semantic memory, persistent user/interlocutor profiles, relations and retrieval.
- **Reasoning and planning:** intention extraction, causal and temporal reasoning, autonomous goals, long-horizon plans and action outcomes.
- **Self-correction:** explicit user corrections become behavioural evidence; complex tensions can be grouped into competing corrective strategies and evaluated over time.
- **Voice:** Whisper or faster-whisper speech recognition, configurable TTS providers and interruptible audio conversation.
- **Vision:** camera capture, image analysis, face recognition, vision memory and optional multimodal routing.
- **Cognitive workspace:** background signals from attention, curiosity, affect, identity, memory, planning and orchestration.
- **Observability:** cognitive health, traces, subsystem state and a live 3D organic sculpture representing current activity.
- **Local-first operation:** designed to run on the local machine and connect to local LM Studio or Ollama servers.

## Current status

The cognitive organism is an active research prototype. Some subsystems are mature enough for daily testing; others are deliberately exposed so that their limits can be measured. The interface and internal architecture are evolving together.

### Research maturity

The current system is strongest as an integrated experimental platform: it already connects more cognitive mechanisms than a conventional chat interface, while remaining small enough to inspect and modify locally. It is not yet a scientifically validated model of consciousness, general intelligence or human-level reasoning. The next improvements should come from benchmark fixtures, longitudinal conversation tests, ablations, latency measurements and independent tester reports.

## Installation

### Linux

```bash
chmod +x setup.sh
./setup.sh
```

### Windows

Run `start.bat`. The script creates the Python virtual environment, installs Python dependencies, installs optional face-recognition support when possible, installs frontend dependencies, builds the React interface and starts the server.

The new interface is served at `http://localhost:8080/next/`.

The frontend is intentionally built during installation. A published checkout needs `frontend/src`, `frontend/public`, `frontend/package.json` and `frontend/package-lock.json`; `frontend/node_modules` and `frontend/dist` are generated and should not be committed.

## Local model setup

The organism works best for local testing with an OpenAI-compatible server such as LM Studio or Ollama. Configure the provider, model and base URL in the settings page or `config.json`. Keep API keys in `.env` or local configuration; never commit them.

### Tested LM Studio configuration

In current local testing, the strongest results have been observed with:

- **Server:** LM Studio with its OpenAI-compatible API enabled;
- **Model:** `gemma-4-e4b-uncensored-hauhaucs-aggressive`;
- **Inference:** reasoning enabled in LM Studio.

This is an empirical compatibility note, not a benchmark claim. Results may vary with the LM Studio version, prompt context, context length, inference settings and available hardware. When reporting a result, include whether reasoning was enabled and the exact model identifier so other testers can reproduce the comparison.

## Testing

Python tests are in `tests/`. Frontend tests are in `frontend/tests/`.

```bash
python -m unittest discover -s tests -v
cd frontend
npm test
```

Some optional tests require the full virtual environment and external services. A test report should mention which providers, models and optional hardware were available.

## Help advance the research

New research testers are welcome, especially people who can evaluate one of these areas:

- multilingual conversation and response-length modes;
- LM Studio, Ollama and other OpenAI-compatible providers;
- audio latency, barge-in, STT and TTS provider selection;
- camera devices, face recognition and multimodal vision models;
- memory recall, user profiles and correction learning;
- autonomous goals, planning, causal reasoning and temporal understanding;
- accessibility, privacy, installation and failure recovery.

You do not need to accept the project’s claims. The most useful testers try to falsify them: repeat a task, inspect the trace, restart the system, change the local model, introduce a correction and check whether later behaviour actually changes. This is how the cognitive organism becomes better than a compelling demo.

Useful tester reports include the operating system, Python and Node versions, provider/model, exact steps, expected result, observed result and a redacted log excerpt. Do not include API keys, face images or private conversation data.

## Privacy and safety

The organism can process text, microphone input, camera frames and face data. Run it only on a trusted machine, keep LAN exposure disabled unless deliberately needed, and review the local data directories before sharing diagnostics. The project is experimental and should not be used as the sole basis for medical, legal, safety-critical or personal security decisions.

## License

Add the project license before public release. Until then, treat this repository as source-available experimental software rather than assuming that redistribution terms are defined.
