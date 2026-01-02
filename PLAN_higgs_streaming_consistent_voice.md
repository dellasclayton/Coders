# Plan: Higgs Audio Streaming with Consistent Voice

## Overview

Update `HiggsAudioServeEngine.generate_delta_stream` to produce consistent voice output using the speaker profile text description method demonstrated in `generation.py`. The key insight from `generation.py` is that voice continuity is maintained by accumulating generated audio tokens and messages as context for subsequent generations.

---

## Key Patterns from generation.py

### 1. Voice Continuity Mechanism (lines 287-370)

```python
# Track accumulated audio tokens and messages
generated_audio_ids = []
generation_messages = []

for idx, chunk_text in enumerate(chunked_text):
    # Add current user message
    generation_messages.append(Message(role="user", content=chunk_text))

    # Build full context: initial messages + accumulated generation messages
    chatml_sample = ChatMLSample(messages=messages + generation_messages)

    # Build audio context: initial audio_ids + all generated audio so far
    context_audio_ids = audio_ids + generated_audio_ids

    # ... prepare sample with context_audio_ids ...
    # ... generate ...

    # Accumulate generated audio for next iteration
    generated_audio_ids.append(audio_out_ids)

    # Add assistant response message
    generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))

    # Apply rolling buffer to cap context length
    if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
        generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
        generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]
```

### 2. System Message for Speaker Profile (user-provided placeholder)

```python
Message(
    role="system",
    content=f"Generate audio following instructions.\n\n<|scene_desc_start|>\n{scene_prompt}\n\n"
            + "\n".join(speaker_desc)
            + "\n<|scene_desc_end|>"
)
```

### 3. Audio Context Preparation (lines 307-324)

```python
context_audio_ids = audio_ids + generated_audio_ids

curr_sample = ChatMLDatasetSample(
    input_ids=torch.LongTensor(input_tokens),
    label_ids=None,
    audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
        if context_audio_ids else None,
    audio_ids_start=torch.cumsum(
        torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
    ) if context_audio_ids else None,
    # ... rest of fields ...
)
```

---

## Implementation Plan

### Step 1: Add New Method `generate_delta_stream_with_context`

Create a new streaming method that accepts context for voice continuity. This method will be the primary interface for consistent voice streaming.

**New method signature:**

```python
async def generate_delta_stream_with_context(
    self,
    messages: List[Message],                    # Initial context messages (includes system message)
    audio_ids: List[torch.Tensor],              # Initial reference audio tokens (empty for profile-only)
    text: str,                                  # Current text chunk to synthesize
    generated_audio_ids: List[torch.Tensor],    # Previously generated audio tokens for context
    generation_messages: List[Message],         # Previous user/assistant pairs for context
    generation_chunk_buffer_size: int = 2,      # Rolling buffer size
    max_new_tokens: int = 2048,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 0.95,
    ras_win_len: Optional[int] = 7,
    ras_win_max_num_repeat: int = 2,
    seed: Optional[int] = None,
    stop_strings: Optional[List[str]] = None,
):
```

### Step 2: Implement Context Building Logic

Following `generation.py` exactly:

```python
# Build generation messages with current text
current_generation_messages = generation_messages.copy()
current_generation_messages.append(Message(role="user", content=text))

# Build full ChatML sample
chatml_sample = ChatMLSample(messages=messages + current_generation_messages)

# Build audio context
context_audio_ids = audio_ids + generated_audio_ids
```

### Step 3: Implement Custom Input Preparation

Create `_prepare_inputs_with_audio_context` method that handles the audio context properly (mirroring generation.py's approach):

```python
def _prepare_inputs_with_audio_context(
    self,
    chat_ml_sample: ChatMLSample,
    context_audio_ids: List[torch.Tensor],
    force_audio_gen: bool = False
):
    input_tokens, _, _, _ = prepare_chatml_sample(chat_ml_sample, self.tokenizer)

    postfix = "<|start_header_id|>assistant<|end_header_id|>\n\n"
    if force_audio_gen:
        postfix += "<|audio_out_bos|>"
    postfix = self.tokenizer.encode(postfix, add_special_tokens=False)
    input_tokens.extend(postfix)

    # Build audio context the same way as generation.py
    if context_audio_ids:
        audio_ids_concat = torch.concat([ele.cpu() for ele in context_audio_ids], dim=1)
        audio_ids_start = torch.cumsum(
            torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0
        )
    else:
        audio_ids_concat = None
        audio_ids_start = None

    sample = ChatMLDatasetSample(
        input_ids=torch.LongTensor(input_tokens),
        label_ids=None,
        audio_ids_concat=audio_ids_concat,
        audio_ids_start=audio_ids_start,
        audio_waveforms_concat=None,
        audio_waveforms_start=None,
        audio_sample_rate=None,
        audio_speaker_indices=None,
    )

    data = self.collator([sample])
    inputs = asdict(data)
    for k, v in inputs.items():
        if isinstance(v, torch.Tensor):
            inputs[k] = v.contiguous().to(self.device)

    return inputs
```

### Step 4: Modify Streamer to Accumulate Audio Tokens

Update `AsyncHiggsAudioStreamer` to collect all audio tokens for return:

```python
class AsyncHiggsAudioStreamer(BaseStreamer):
    def __init__(self, ...):
        # ... existing code ...
        self.accumulated_audio_tokens = []  # Collect audio tokens for context
```

Or alternatively, accumulate tokens in `generate_delta_stream_with_context` as they're yielded.

### Step 5: Return Context for Next Generation

The method should yield deltas AND return the accumulated audio tokens after streaming completes. One approach:

```python
async def generate_delta_stream_with_context(...):
    # ... setup and streaming ...

    audio_tokens_list = []
    async for delta in streamer:
        if delta.audio_tokens is not None:
            audio_tokens_list.append(delta.audio_tokens)
        yield delta

    # After streaming completes, process the accumulated tokens
    if audio_tokens_list:
        audio_out_ids = torch.stack(audio_tokens_list, dim=1)  # Shape: [num_codebooks, seq_len]
        if self.model.config.use_delay_pattern:
            audio_out_ids = revert_delay_pattern(audio_out_ids)
        audio_out_ids = audio_out_ids.clip(0, self.audio_codebook_size - 1)[:, 1:-1]

        # Yield final delta with accumulated audio_out_ids for context
        yield HiggsAudioStreamerDelta(
            finish_reason="stop",
            audio_tokens=audio_out_ids,  # Full accumulated tokens for next iteration
        )
```

### Step 6: Apply Rolling Buffer

After generation completes, caller applies buffer logic:

```python
# (Called by the FastAPI layer after each generation)
generated_audio_ids.append(audio_out_ids)
generation_messages.append(Message(role="assistant", content=AudioContent(audio_url="")))

if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
    generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
    generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]
```

---

## Required Imports

Add to `serve_engine.py`:

```python
from ..data_types import Message, AudioContent
```

---

## Data Flow Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Layer (manages state across requests)                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Persistent state per session:                                              │
│    - messages: [system_message]  (includes speaker profile)                 │
│    - audio_ids: []               (empty for profile-only mode)              │
│    - generated_audio_ids: []     (accumulated from generations)             │
│    - generation_messages: []     (user/assistant pairs)                     │
│    - generation_chunk_buffer_size: 2                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ generate_delta_stream_with_context(                                         │
│   messages,                         # Initial context                       │
│   audio_ids,                        # Reference audio (empty list)          │
│   text,                             # Current chunk to synthesize           │
│   generated_audio_ids,              # Previous generated audio              │
│   generation_messages,              # Previous user/assistant pairs         │
│   generation_chunk_buffer_size,     # Buffer size                           │
│   ...                                                                       │
│ )                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Yields HiggsAudioStreamerDelta:                                             │
│   - audio_tokens (per-step for streaming audio)                             │
│   - text_tokens (if any)                                                    │
│   - finish_reason="stop" with accumulated audio_out_ids (final delta)       │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Layer updates state:                                                │
│   - generated_audio_ids.append(audio_out_ids)                               │
│   - generation_messages.append(user_msg)                                    │
│   - generation_messages.append(assistant_msg)                               │
│   - Apply rolling buffer                                                    │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Files to Modify

1. **`backend/boson_multimodal/serve/serve_engine.py`**
   - Add `_prepare_inputs_with_audio_context` method
   - Add `generate_delta_stream_with_context` method
   - Add import for `Message`, `AudioContent` from `..data_types`

---

## Variables and Naming (matching generation.py)

| generation.py Variable | serve_engine.py Variable | Purpose |
|------------------------|--------------------------|---------|
| `messages` | `messages` | Initial context (system message with speaker profile) |
| `audio_ids` | `audio_ids` | Reference audio tokens (empty list for profile-only) |
| `generated_audio_ids` | `generated_audio_ids` | Accumulated generated audio tokens |
| `generation_messages` | `generation_messages` | User/assistant message pairs |
| `context_audio_ids` | `context_audio_ids` | `audio_ids + generated_audio_ids` |
| `generation_chunk_buffer_size` | `generation_chunk_buffer_size` | Rolling buffer size |
| `audio_out_ids` | `audio_out_ids` | Current generation's output tokens |

---

## System Message Placeholder

The system message will be constructed by the FastAPI layer and passed as part of `messages`:

```python
Message(
    role="system",
    content=f"Generate audio following instructions.\n\n<|scene_desc_start|>\n{scene_prompt}\n\n"
            + "\n".join(speaker_desc)
            + "\n<|scene_desc_end|>"
)
```

This will be wired in during the FastAPI integration task.

---

## Testing Strategy

1. **Unit test**: Call `generate_delta_stream_with_context` with empty context, verify output
2. **Context test**: Call twice in sequence, passing accumulated context, verify voice consistency
3. **Buffer test**: Exceed buffer size, verify old context is pruned correctly

---

## Open Questions for Review

1. **Thread safety**: The current implementation uses threading for generation. Should we add any locking for the context state?

2. **Audio token format**: The streamer yields individual audio tokens. For context, we need the full `audio_out_ids` tensor. Should this be accumulated in the streamer or in the method?

3. **Delay pattern handling**: Should `revert_delay_pattern` be applied before storing in `generated_audio_ids`, or during context preparation?
   - Looking at generation.py lines 352-360, it's applied BEFORE appending to `generated_audio_ids`
   - So we should apply it in `generate_delta_stream_with_context` before yielding the final context

---

## Summary

The key insight is that `generation.py` maintains voice consistency by:
1. Building audio context from initial audio_ids + all previously generated audio
2. Building message context from initial messages + all user/assistant pairs
3. Using a rolling buffer to cap context length

We will replicate this exactly in `generate_delta_stream_with_context`, allowing the FastAPI layer to manage the persistent state between streaming calls.
