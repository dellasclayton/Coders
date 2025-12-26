# Project Overview

We are creating a low latency real time streaming AI voice chat that runs STT, LLM, and TTS on a remote/cloud GPU.
Backend is Python FastAPI server that connects to the client/browser via a single WebSocket.

## Code Guidelines

- Prefer clean, manageable, human readable coding style. Please do not create complex over-engineered solutions.
- Code components should be modular and easily integrated.
- Functions and variables should be descriptive and human readable for user to understand.
- Always plan for backend/frontend integration.
- Helpful/useful commenting is highly encouraged.
- Complete only the tasks you are presented with. No add-ons.
- Ask questions, don't assume.

### Python (Backend)

- FastAPI server with browser/client.
- WebSocket handles both binary (audio) and JSON (command) messages.
- Supabase used for database.
- Speech to Text uses RealtimeSTT (faster whisper) for real-time transcription and callbacks.
- LLM/Chat via OpenRouter API with streaming response.
- Text to Speech uses Higgs Audio - HiggsAudioServeEngine streams PCM chunks to browser/client for playback.

### JavaScript (frontend)

- Use a procedural/functional style, not object-oriented.
- No ES6 classes - use plain functions and variables instead.
- Self-contained single file approach.
- Direct DOM manipulation. Use document.getElementById(), .addEventListener() directly.
- Clear structure: variables at top, functions in the middle, initialization at bottom.
- Linear flow - code should read top-to-bottom, naturally.
- No dependency injection - don't pass managers or require external instantiation.
- Simple state management - use plain variables, not this.property

## Notable Features

- Concurrent synthesis, start TTS audio generation as soon as first chunk from response is available, concurrent generation + playback.
- Proper queue management, prioritization + buffering, optimized for multiple chunks.
- Can handle multiple character chat with speaker tracking and next character.

**Important** Must plan for automatic listening/recording. User presses voiceButton (index.html) to start_listening, then _on_audio_stream_start we stop_listening, and _on_audio_stream_stop we start listening again. Please include in appropriate file.

## Planned File Structure

Base/
├── backend/
│   ├── RealtimeSTT/
│   │   ├── audio_input.py
│   │   └── audio_recorder.py
│   ├── RealtimeTTS/
│   │   ├── text_to_stream.py
│   │   └── threadsafe_generators.py
│   ├── bosun_multimodal/
│   │   └── examples/
│   │   |    └── generation.py
│   |   └── serve/
│   │   |    └── serve_engine.py
│   └── server.py
└── frontend/
    ├── css/
    │   └── styles.css
    ├── js/
    │   ├── app.js
    │   ├── audio-capture.js
    │   ├── audio-player.js
    │   ├── characters.js
    │   ├── chatui.js
    │   └── websocket.js
    └── index.html

---

# Real-Time Streaming TTS with Higgs Audio - Implementation Plan

## Overview
Integrate Higgs Audio TTS with OpenRouter LLM streaming to provide concurrent synthesis and playback with voice consistency.

## Architecture Flow
```
OpenRouter LLM Stream → Sentence Buffer → TTS Pipeline → Audio Decoder → WebRTC → Browser Playback
                                              ↓
                                    Context Manager (voice consistency)
```

## Step-by-Step Implementation

### Step 1: Create TTS Service Module (`backend/tts_service.py`)
**Purpose**: Core TTS engine wrapper with context management

**Key Components**:
- Initialize `HiggsAudioServeEngine` with model and audio tokenizer
- Create async generator wrapper for streaming synthesis
- Implement context management for voice consistency (accumulated `generated_audio_ids`)
- Handle audio token buffering and PCM conversion

**Implementation Details**:
- Load model: `bosonai/higgs-audio-v2-generation-3B-base`
- Load tokenizer: `bosonai/higgs-audio-v2-tokenizer`
- Maintain `generated_audio_ids` buffer (last 3-5 chunks)
- Provide methods: `initialize()`, `synthesize_stream()`, `clear_context()`

---

### Step 2: Implement Sentence Buffer/Accumulator (`backend/text_buffer.py`)
**Purpose**: Accumulate LLM tokens into complete sentences for TTS

**Key Components**:
- Accumulate LLM tokens into sentences using punctuation detection (`.`, `!`, `?`)
- Yield complete sentences for TTS synthesis
- Handle edge cases (abbreviations, numbers, incomplete sentences)

**Implementation Details**:
- Stream tokens from OpenRouter
- Detect sentence boundaries with regex patterns
- Handle special cases: "Dr.", "Mr.", "Mrs.", etc.
- Minimum/maximum sentence length thresholds
- Flush remaining text on stream end

---

### Step 3: Create Concurrent Synthesis Pipeline (`backend/synthesis_pipeline.py`)
**Purpose**: Manage multiple concurrent TTS synthesis tasks

**Key Components**:
- Queue-based architecture for managing multiple concurrent synthesis tasks
- Process first sentence immediately when available
- Queue subsequent sentences for concurrent processing
- Maintain FIFO order for playback

**Implementation Details**:
- Use `asyncio.Queue` for sentence queue
- Spawn concurrent synthesis tasks with `asyncio.create_task()`
- Track task completion order
- Limit concurrent tasks (e.g., max 3 simultaneous)
- Handle errors gracefully without blocking pipeline

---

### Step 4: Implement Audio Chunk Streamer (`backend/audio_streamer.py`)
**Purpose**: Convert audio tokens to PCM and stream to client

**Key Components**:
- Collect audio tokens from `generate_delta_stream`
- Batch tokens (50-100 tokens = 1-2 seconds audio)
- Decode to PCM using audio tokenizer
- Apply hamming window for smooth transitions
- Stream PCM chunks to WebRTC connection

**Implementation Details**:
```python
# Pseudo-code structure
async def stream_audio_chunks(delta_generator, context_audio_ids):
    audio_token_buffer = []
    async for delta in delta_generator:
        if delta.audio_tokens:
            audio_token_buffer.append(delta.audio_tokens)

            if len(audio_token_buffer) >= CHUNK_SIZE:
                # Stack tokens and decode
                vq_code = torch.stack(audio_token_buffer, dim=1)
                vq_code = revert_delay_pattern(vq_code).clip(0, codebook_size-1)[:, 1:-1]
                pcm_audio = audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                # Store for context
                context_audio_ids.append(vq_code)

                # Stream to client
                yield pcm_audio
                audio_token_buffer = []
```

**Technical Parameters**:
- Sampling rate: 24000 Hz
- Chunk size: 50-100 tokens (1-2 seconds)
- Audio format: PCM, 16-bit signed integer, mono
- Hamming window: 960 samples for smooth transitions

---

### Step 5: Add Character Profile Management (`backend/character_profiles.py`)
**Purpose**: Manage speaker profiles and scene descriptions

**Key Components**:
- Store speaker descriptions (text-based profiles)
- Manage scene prompts for acoustic environment
- Build ChatMLSample messages with proper formatting
- Support multiple speakers with `[SPEAKER0]`, `[SPEAKER1]` tags

**Implementation Details**:
```python
# Character profile structure
character_profile = {
    "speaker_id": "SPEAKER0",
    "description": "feminine, young adult, warm, conversational",
    "scene_prompt": "quiet indoor environment with minimal background noise"
}

# Build ChatMLSample
def build_chat_sample(text, profile, conversation_history):
    system_message = Message(
        role="system",
        content=(
            "Generate audio following instruction.\n\n"
            "<|scene_desc_start|>\n"
            f"{profile['scene_prompt']}\n\n"
            f"{profile['speaker_id']}: {profile['description']}\n"
            "<|scene_desc_end|>"
        )
    )

    messages = [system_message] + conversation_history + [
        Message(role="user", content=f"[{profile['speaker_id']}] {text}")
    ]

    return ChatMLSample(messages=messages)
```

**Profile Examples**:
- Character 1: "feminine, young adult, energetic, cheerful"
- Character 2: "masculine, middle-aged, calm, authoritative"
- Scene: "quiet indoor studio environment"

---

### Step 6: Integrate with Server WebRTC Handler (`backend/server.py`)
**Purpose**: Connect all components and handle WebRTC communication

**Key Components**:
- Add endpoint/message handler for TTS requests
- Connect LLM streaming → sentence buffer → TTS pipeline
- Stream PCM audio chunks to browser via WebRTC data channel
- Handle stop/interrupt signals

**Implementation Flow**:
```python
async def handle_chat_message(message, webrtc_connection):
    # 1. Stream from OpenRouter
    llm_stream = openrouter.stream(message)

    # 2. Buffer into sentences
    sentence_buffer = SentenceBuffer()

    # 3. Process sentences concurrently
    async for sentence in sentence_buffer.process(llm_stream):
        # 4. Synthesize with TTS
        async for pcm_chunk in tts_service.synthesize_stream(
            text=sentence,
            character_profile=current_character,
            context_audio_ids=context_manager.get_context()
        ):
            # 5. Send to browser
            await webrtc_connection.send_audio(pcm_chunk)
```

**WebRTC Message Types**:
- `audio_stream_start`: Notify client audio is starting
- `audio_chunk`: Binary PCM data
- `audio_stream_stop`: Notify client audio is complete
- `tts_interrupt`: Stop current synthesis

---

### Step 7: Update Frontend Audio Player (`frontend/js/audio-player.js`)
**Purpose**: Receive and play PCM audio chunks seamlessly

**Key Components**:
- Receive PCM chunks from WebRTC (24kHz, 16-bit, mono)
- Buffer chunks in AudioContext queue
- Play immediately when first chunk arrives
- Continue playing subsequent chunks seamlessly

**Implementation Details**:
```javascript
// Audio player state
let audioContext = null;
let audioQueue = [];
let isPlaying = false;
let nextStartTime = 0;

async function handleAudioChunk(pcmData) {
    // Convert PCM bytes to Float32Array
    const audioBuffer = await pcmToAudioBuffer(pcmData, 24000);

    // Add to queue
    audioQueue.push(audioBuffer);

    // Start playing if not already
    if (!isPlaying) {
        playNextChunk();
    }
}

function playNextChunk() {
    if (audioQueue.length === 0) {
        isPlaying = false;
        return;
    }

    isPlaying = true;
    const buffer = audioQueue.shift();
    const source = audioContext.createBufferSource();
    source.buffer = buffer;
    source.connect(audioContext.destination);

    // Schedule seamless playback
    const startTime = Math.max(nextStartTime, audioContext.currentTime);
    source.start(startTime);
    nextStartTime = startTime + buffer.duration;

    // Play next when this ends
    source.onended = () => playNextChunk();
}
```

**Technical Requirements**:
- Use Web Audio API AudioContext
- Convert PCM to AudioBuffer
- Schedule sources for gapless playback
- Handle buffer underruns gracefully

---

### Step 8: Implement Voice Consistency System
**Purpose**: Maintain consistent voice characteristics across conversation

**Key Components**:
- Maintain `generated_audio_ids` buffer across conversation
- Include last N chunks as context (buffer size: 3-5 chunks)
- Prepend context to each new synthesis request
- Clear context on character change or conversation reset

**Implementation Details**:
```python
class VoiceContextManager:
    def __init__(self, max_chunks=5):
        self.max_chunks = max_chunks
        self.context_audio_ids = []
        self.current_character = None

    def add_generated_audio(self, audio_ids):
        """Add newly generated audio to context"""
        self.context_audio_ids.append(audio_ids)

        # Limit buffer size
        if len(self.context_audio_ids) > self.max_chunks:
            self.context_audio_ids = self.context_audio_ids[-self.max_chunks:]

    def get_context(self):
        """Get context for next synthesis"""
        if not self.context_audio_ids:
            return None
        return torch.cat(self.context_audio_ids, dim=1)

    def clear_context(self):
        """Clear context (on character change or reset)"""
        self.context_audio_ids = []
```

**Context Usage**:
- Prepend to ChatMLDatasetSample as `audio_ids_concat` and `audio_ids_start`
- Model uses context to maintain voice characteristics
- Balance between consistency and memory usage

---

### Step 9: Add Automatic Listen/Stop Logic (`frontend/js/app.js`)
**Purpose**: Automatically manage STT recording based on TTS playback

**Key Components**:
- On `audio_stream_start`: stop listening (stop STT recording)
- On `audio_stream_stop`: resume listening (restart STT recording)
- Update UI to reflect state changes

**Implementation Details**:
```javascript
// Listen for TTS events
webrtc.on('audio_stream_start', () => {
    // Stop STT recording while AI is speaking
    if (isRecording) {
        stopRecording();
        autoStoppedForTTS = true;
    }
    updateUIState('ai-speaking');
});

webrtc.on('audio_stream_stop', () => {
    // Resume STT recording after AI finishes
    if (autoStoppedForTTS) {
        startRecording();
        autoStoppedForTTS = false;
    }
    updateUIState('listening');
});

// Manual voice button override
voiceButton.addEventListener('click', () => {
    if (isRecording) {
        stopRecording();
        autoStoppedForTTS = false; // User manually stopped
    } else {
        startRecording();
    }
});
```

**State Management**:
- Track whether STT was auto-stopped vs. manually stopped
- Only auto-resume if auto-stopped (respect user manual control)
- Clear visual feedback for current state

---

### Step 10: Testing & Optimization
**Purpose**: Ensure quality, performance, and reliability

**Test Cases**:
1. **End-to-end latency**: Measure time from LLM first token to first audio playback (target: <2 seconds)
2. **Voice consistency**: Verify voice characteristics remain stable across multiple turns
3. **Concurrent synthesis**: Test multiple sentences being synthesized simultaneously
4. **Buffer underruns**: Handle slow synthesis or network issues gracefully
5. **Character switching**: Ensure context clears properly when changing speakers
6. **Long responses**: Test with multi-paragraph responses (chunking)
7. **Interruption handling**: Stop TTS mid-sentence when user interrupts
8. **Error recovery**: Handle model errors, network failures, etc.

**Optimization Targets**:
- Buffer size tuning: Balance latency vs. quality
- Context window size: Balance consistency vs. memory
- Concurrent task limit: Balance throughput vs. GPU memory
- Sentence detection: Minimize false positives/negatives

**Performance Metrics**:
- Time to first audio: <2 seconds
- Audio quality: MOS score >4.0
- Voice consistency: Subjective evaluation across 10+ turns
- CPU/GPU utilization: <80% sustained load
- Memory usage: <8GB VRAM for TTS model

---

## Key Technical Specifications

### Audio Format
- **Codec**: PCM (uncompressed)
- **Sampling Rate**: 24000 Hz
- **Bit Depth**: 16-bit signed integer
- **Channels**: Mono
- **Chunk Size**: 50-100 tokens (1-2 seconds of audio)

### Model Parameters
- **Model**: `bosonai/higgs-audio-v2-generation-3B-base`
- **Audio Tokenizer**: `bosonai/higgs-audio-v2-tokenizer`
- **Temperature**: 0.7 (adjustable for voice variation)
- **Top-k**: 50
- **Top-p**: 0.95
- **Max New Tokens**: 2048
- **RAS Window**: 7 (repetition aware sampling)

### Context Management
- **Context Window**: Last 3-5 generated audio chunks
- **Buffer Format**: Concatenated audio token IDs
- **Clear Triggers**: Character change, conversation reset, manual clear

### Profile Format
```yaml
speaker_id: SPEAKER0
description: feminine, young adult, warm, conversational, clear pronunciation
scene_prompt: quiet indoor environment with minimal background noise
```

### Integration Points
- **LLM**: OpenRouter API with streaming
- **Backend**: FastAPI with WebRTC
- **Transport**: WebRTC data channels (binary PCM chunks)
- **Frontend**: Web Audio API for playback
- **STT**: RealtimeSTT (auto-stop during TTS playback)

---

## Implementation Notes

### Voice Profile Best Practices
- Use descriptive adjectives: age, gender, tone, emotion, clarity
- Keep descriptions concise (5-10 words)
- Test different combinations for desired characteristics
- Scene prompts improve quality significantly

### Latency Optimization
- Start TTS on first complete sentence (don't wait for full response)
- Use smaller chunk sizes for lower latency (trade-off: quality)
- Decode audio in batches of 50-100 tokens (sweet spot)
- Prefetch and buffer 1-2 chunks ahead of playback

### Error Handling
- Gracefully handle model generation failures
- Fallback to text-only response if TTS fails
- Implement retry logic with exponential backoff
- Log errors for monitoring and debugging

### Memory Management
- Limit context buffer to prevent memory growth
- Clear KV cache between conversations
- Use StaticCache with appropriate sizes
- Monitor GPU memory usage and adjust buffers

---
