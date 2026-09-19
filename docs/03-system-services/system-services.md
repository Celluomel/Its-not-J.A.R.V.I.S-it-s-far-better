# System Services Documentation

## 📊 Overview: Managers Subsystem (6 Core Service Modules)

The managers subsystem orchestrates external interfaces, handles multi-session coordination, enforces security boundaries, and manages channel communications for the cognitive organism.

---

## A. LLM Manager (`managers/llm_manager.py`)

### Purpose
Central interface and scheduling hub for all Large Language Model interactions within PandoraBOX.

### Key Responsibilities:

#### 1. Model Routing & Load Balancing
```python
# Routes prompts to appropriate models based on:
- Task complexity (simple routing → complex reasoning)
- Model availability and load
- User preferences and cost considerations
- Latency requirements
```

**Supported Models:**
- Local LM Studio deployment (`custom-localhost-1234/qwen/qwen3.5-9b`)
- Cloud APIs (fallback options for reliability)

#### 2. Prompt Construction & Templating
- Dynamic prompt assembly based on cognitive context
- Context window management and optimization  
- System instruction injection for task-specific behavior
- Multi-turn conversation state preservation

#### 3. Response Streaming & Handling
- Handles streaming token generation from models
- Processes tool call requests within LLM responses
- Manages error recovery for failed completions
- Implements retry logic with backoff strategies

### Integration Points:
```
┌─────────────┐
│ UI Action   │ → prompts LLM manager for task handling
└─────────────┘
      ↓
┌─────────────────────────────────┐
│   LLM Manager                    │
│  • Routes to appropriate model   │
│  • Constructs optimized prompt   │
│  • Streams response tokens       │
│  • Parses tool calls if needed   │
│  • Handles errors and retries    │
└─────────────────────────────────┘
      ↓
┌──────────────┐
│ Cognitive    │ ← receives LLM-generated reasoning
│ Modules      │
└──────────────┘
```

---

## B. Audio Manager (`managers/audio_manager.py`)

### Purpose
Handles all voice/speech interface functionality including text-to-speech output and speech recognition input.

### Key Features:

#### 1. Text-to-Speech (TTS) Generation
- Converts cognitive outputs to spoken audio
- Supports multiple voice profiles (preference system for voice selection)
- Handles streaming audio playback

#### 2. Speech Recognition (STT) Input  
- Captures microphone input for voice commands
- Processes speech-to-text transcripts
- Triggers appropriate cognitive events from voice input

#### 3. Audio Device Management
- Enumerates available audio/speech devices
- Manages device connections and disconnections
- Handles audio playback preferences

### Integration:
```python
# Called when UI requests audio output
audio_manager.synthesize_speech(text="Cognitive insight...")

# Called for voice input processing  
stt_result = audio_manager.transcribe_audio(audio_stream)
```

---

## C. Vision Manager (`managers/vision_manager.py`)

### Purpose
Processes camera input and visual feature extraction from environmental sensors.

### Key Functions:

#### 1. Camera Capture & Frame Processing
- Captures frames from connected camera devices
- Preprocesses frames (resizing, normalization, augmentation)
- Handles multiple camera sources simultaneously

#### 2. Visual Feature Extraction  
- Detects motion patterns and changes
- Identifies objects/regions of interest
- Extracts visual features for semantic processing

#### 3. Vision Event Publishing
```python
# Publishes vision events to cognitive event bus
vision_event = {
    'type': 'PERCEPTION_UPDATE',
    'modality': 'VISUAL', 
    'features': extracted_visual_features,
    'timestamp': time.time()
}
cognitive_event_bus.publish(vision_event)
```

---

## D. Session Manager (`managers/session_manager.py`)

### Purpose
Coordinates multiple concurrent user sessions and maintains their individual states within a single cognitive organism instance.

### Key Features:

#### 1. Multi-Session State Tracking
- Maintains separate working memories per session
- Isolates goals, context, and identity aspects between users
- Prevents cross-session interference

#### 2. Session Lifecycle Management
- Handles session initialization and shutdown
- Manages resource cleanup between sessions
- Tracks active/inactive/terminated sessions

#### 3. Session Context Sharing (Selective)
```python
# Can share knowledge across sessions when appropriate:
session_manager.share_with_other_sessions(context={'concept': 'knowledge'}, 
                                             target_sessions=['session_b', 'session_c'])
```

### Architecture:
```
┌─────────────────────────────────────────────┐
│   Single Cognitive Organism Instance        │
│  (Shared knowledge, persistent memory)      │
└─────────────────────────────────────────────┘
          ↓ shares resources
┌───────────────┐  ┌───────────────┐  ┌───────────────┐
│ Session A     │  │ Session B     │  │ Session C     │
│ User #1       │  │ User #2       │  │ User #3       │
│ Isolated      │  │ Isolated      │  │ Isolated      │
│ Goals/Context │  │ Goals/Context │  │ Goals/Context │
└───────────────┘  └───────────────┘  └───────────────┘
```

---

## E. Security Manager (`managers/security_manager.py`)

### Purpose
Enforces content filters, access controls, and safety boundaries to prevent harmful operations.

### Key Features:

#### 1. Content Filtering
- Scans incoming user input for malicious patterns
- Filters LLM outputs for safety compliance
- Blocks requests that violate usage policies

#### 2. Access Control Enforcement
```python
# Checks permissions before allowing sensitive operations
if security_manager.check_permission('execute_code'):
    allow_execution = True
else:
    deny_and_explain()
```

#### 3. Safety Boundary Monitoring
- Monitors for potential harm patterns
- Enforces ethical AI guidelines  
- Escalates concerning behaviors to safety system

### Integration Layer:
All cognitive modules must pass through security manager before executing potentially harmful actions:
```python
# Pattern used throughout system:
action_allowed = security_manager.validate_action(intended_action, 
                                                    context_state)
if action_allowed:
    execute_action()
else:
    trigger_safety_response()
```

---

## F. Messaging Manager (`managers/messaging_manager.py`)

### Purpose
Handles integrations with external communication channels (Telegram, Discord, Signal, etc.)

### Channel-Specific Handlers:

#### 1. Telegram Integration
- Sends/receives messages in configured channels
- Manages bot presence and authentication
- Handles message effects (e.g., invisible ink)

#### 2. Discord Integration  
- Posts to Discord channels/groups
- Supports thread creation for ACP sessions
- Manages webhook integrations

#### 3. Signal Integration
- Sends encrypted messages via Signal Protocol
- Authenticates with Signal users
- Handles contact synchronization

### Message Routing:
```python
# Unified interface for different channels
messaging_manager.send_to_channel(
    channel='telegram' or 'discord' or 'signal',
    message="Cognitive insight...",
    metadata={...}  # Optional attachment, formatting, etc.
)
```

---

## 🔄 System Service Coordination

### Inter-Manager Communication:
Managers coordinate through shared event bus and state:

```python
# Example: Handling voice-activated camera query
# 1. Audio manager captures "Look at this" command
audio_manager.listen_microphone() → speech_recognition_result

# 2. Security manager validates the request
security_manager.validate_request(recognized_command)

# 3. Vision manager captures relevant frame  
vision_manager.capture_frame(camera_id='main')

# 4. LLM manager processes context and generates response
llm_manager.generate_response(command, visual_features)

# 5. Messaging manager delivers output to user channel
messaging_manager.send_to_channel(user_channels, response_text + attachment)
```

---

## 📊 Service Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              PandoraBOX System Services Layer                    │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌─────────────────┐   │
│  │ LLM Manager  │→ │ Audio Mgr    │←→│ Vision Manager  │   │
│  │ (Prompting)  │  │ (Voice I/O)  │  │ (Camera Input)  │   │
│  └──────────────┘  └──────────────┘  └─────────────────┘   │
│                    ↓                                       │
│  ┌──────────────────────────────────────────────────┐      │
│  │     Security Manager (Central Gatekeeper)        │      │
│  │   • Content Filters    • Access Control Checks   │      │
│  └──────────────────────────────────────────────────┘      │
│                    ↓                                       │
│  ┌──────────────┐  ┌─────────────────────────────┐         │
│  │ Messaging Mgr│→│ Session Manager              │         │
│  │ (Channel IO) │  │ (Multi-session Coordination) │         │
│  └──────────────┘  └─────────────────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Design Principles

1. **Service Abstraction:** Each manager encapsulates specific external interface concerns
2. **Central Security:** Security manager as gatekeeper for all potentially harmful actions
3. **Multi-Session Support:** Session manager enables concurrent user interactions  
4. **Channel Agnosticism:** Messaging manager provides unified interface across platforms
5. **Event Integration:** All managers subscribe to cognitive event bus for coordination

---

## 🔌 External Dependencies Managed

| Manager | External Services | Interface Type |
|---------|-------------------|----------------|
| LLM Manager | LM Studio, Cloud APIs | gRPC/HTTP |
| Audio Manager | ElevenLabs, Whisper | WebSocket/HTTP |
| Vision Manager | Camera devices, OpenCV | Direct hardware |
| Messaging Manager | Telegram/ Discord/Signal | API/Webhooks |
| Session Manager | Local state, database | Internal SQLite |
