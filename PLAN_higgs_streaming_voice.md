# Plan: Higgs Audio Streaming with Consistent Voice via Speaker Profile

## Objective
Enable `HiggsAudioServeEngine.generate_delta_stream()` to produce streaming audio with a consistent voice across multiple generations, using the same context-passing method demonstrated in `HiggsAudioModelClient.generate()` (generation.py).

---

## Key Insight from generation.py

Voice consistency is achieved by **reusing previously generated audio tokens as context**:

```python
# For each new chunk:
context_audio_ids = audio_ids + generated_audio_ids

# Build sample with concatenated context:
curr_sample = ChatMLDatasetSample(
    input_ids=torch.LongTensor(input_tokens),
    audio_ids_concat=torch.concat([ele.cpu() for ele in context_audio_ids], dim=1) if context_audio_ids else None,
    audio_ids_start=torch.cumsum(torch.tensor([0] + [ele.shape[1] for ele in context_audio_ids], dtype=torch.long), dim=0) if context_audio_ids else None,
    ...
)

# After generation, capture output tokens:
audio_out_ids = ele
if self._config.use_delay_pattern:
    audio_out_ids = revert_delay_pattern(audio_out_ids)
audio_out_ids = audio_out_ids.clip(0, self._audio_tokenizer.codebook_size - 1)[:, 1:-1]
generated_audio_ids.append(audio_out_ids)

# Rolling buffer (keep last N chunks):
if generation_chunk_buffer_size is not None and len(generated_audio_ids) > generation_chunk_buffer_size:
    generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
    generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]
```

---

## Current State of generate_delta_stream

**What already works:**
- `_prepare_inputs()` already accepts precomputed audio tokens via `ChatMLSample.misc["audio_ids"]` (lines 295-316)
- The streaming mechanism via `AsyncHiggsAudioStreamer` works

**What's missing:**
- No way to return accumulated audio tokens after streaming completes
- Caller has no guidance on how to process streamed tokens for context reuse

---

## Implementation Plan

### Step 1: Add Return Value for Generated Audio Tokens

Modify `generate_delta_stream()` to yield a final delta containing the processed audio tokens that can be used as context for the next generation.

**Changes to serve_engine.py:**

```python
async def generate_delta_stream(
    self,
    chat_ml_sample: ChatMLSample,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: float = 0.95,
    stop_strings: Optional[List[str]] = None,
    force_audio_gen: bool = False,
    ras_win_len: Optional[int] = 7,
    ras_win_max_num_repeat: int = 2,
    seed: Optional[int] = None,
):
    # ... existing setup code ...

    # Accumulate audio tokens during streaming
    audio_token_frames = []

    async for delta in streamer:
        if delta.audio_tokens is not None:
            audio_token_frames.append(delta.audio_tokens)
        yield delta

    # After streaming completes, process accumulated tokens for context reuse
    if audio_token_frames:
        # Stack frames: each frame is [num_codebooks], result is [num_codebooks, num_frames]
        raw_audio_tokens = torch.stack(audio_token_frames, dim=1)

        # Apply same post-processing as generation.py
        if self.model.config.use_delay_pattern:
            raw_audio_tokens = revert_delay_pattern(raw_audio_tokens)

        # Clip and trim (matching generation.py exactly)
        processed_audio_ids = raw_audio_tokens.clip(0, self.audio_codebook_size - 1)[:, 1:-1]

        # Yield final delta with processed tokens for context reuse
        yield HiggsAudioStreamerDelta(
            finish_reason="stop",
            audio_tokens=processed_audio_ids,  # Shape: [num_codebooks, num_frames]
        )
    else:
        yield HiggsAudioStreamerDelta(finish_reason="stop")
```

### Step 2: Distinguish Raw vs Processed Audio Tokens in Delta

To avoid confusion between streamed raw tokens and the final processed context tokens, we have two options:

**Option A (Recommended):** Add a new field to `HiggsAudioStreamerDelta`:

```python
@dataclass
class HiggsAudioStreamerDelta:
    text: Optional[str] = None
    text_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None
    audio_context_tokens: Optional[torch.Tensor] = None  # NEW: Processed tokens for next-gen context
    finish_reason: Optional[str] = None
```

**Option B:** Use `finish_reason` presence to distinguish (final delta has processed tokens).

### Step 3: Document the Calling Pattern

The caller (e.g., fastserver.py) must manage:

1. **System message with speaker profile** (from `prepare_generation_context`):
```python
system_message = (
    "Generate audio following instruction.\n\n"
    f"<|scene_desc_start|>\n{scene_prompt}\n\n"
    f"SPEAKER0: {speaker_profile_description}\n"
    "<|scene_desc_end|>"
)
```

2. **Context accumulation**:
```python
generated_audio_ids = []  # Accumulates across generations
generation_messages = []  # Tracks user/assistant messages
generation_chunk_buffer_size = 2  # Keep last 2 chunks

async for delta in engine.generate_delta_stream(chat_ml_sample, ...):
    if delta.finish_reason == "stop" and delta.audio_context_tokens is not None:
        # Store for next generation
        generated_audio_ids.append(delta.audio_context_tokens)

        # Rolling buffer
        if len(generated_audio_ids) > generation_chunk_buffer_size:
            generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
            generation_messages = generation_messages[(-2 * generation_chunk_buffer_size):]
    elif delta.audio_tokens is not None:
        # Stream audio to client (raw tokens for real-time playback)
        yield audio_chunk
```

3. **Passing context in subsequent calls**:
```python
chat_ml_sample = ChatMLSample(
    messages=base_messages + generation_messages + [new_user_message],
    misc={"audio_ids": generated_audio_ids}  # Pass accumulated context
)
```

---

## Detailed Code Changes

### File: serve_engine.py

#### Change 1: Add `audio_context_tokens` field to HiggsAudioStreamerDelta

Location: Lines 26-34

```python
@dataclass
class HiggsAudioStreamerDelta:
    """Represents a chunk of generated content, either text or audio tokens."""

    text: Optional[str] = None
    text_tokens: Optional[torch.Tensor] = None
    audio_tokens: Optional[torch.Tensor] = None
    audio_context_tokens: Optional[torch.Tensor] = None  # Processed tokens for context reuse
    finish_reason: Optional[str] = None
```

#### Change 2: Modify generate_delta_stream to accumulate and return processed tokens

Location: Lines 452-515

Replace the existing method with:

```python
async def generate_delta_stream(
    self,
    chat_ml_sample: ChatMLSample,
    max_new_tokens: int,
    temperature: float = 0.7,
    top_k: Optional[int] = None,
    top_p: float = 0.95,
    stop_strings: Optional[List[str]] = None,
    force_audio_gen: bool = False,
    ras_win_len: Optional[int] = 7,
    ras_win_max_num_repeat: int = 2,
    seed: Optional[int] = None,
):
    """
    Generate audio from a chatml sample with streaming output.

    For consistent voice across multiple generations, pass previously generated
    audio tokens via chat_ml_sample.misc["audio_ids"] (list of tensors).

    The final yielded delta (with finish_reason="stop") contains audio_context_tokens
    which should be accumulated and passed back for subsequent generations.

    Example usage for voice consistency:
        generated_audio_ids = []
        generation_chunk_buffer_size = 2

        async for delta in engine.generate_delta_stream(sample, ...):
            if delta.finish_reason == "stop":
                if delta.audio_context_tokens is not None:
                    generated_audio_ids.append(delta.audio_context_tokens)
                    if len(generated_audio_ids) > generation_chunk_buffer_size:
                        generated_audio_ids = generated_audio_ids[-generation_chunk_buffer_size:]
            elif delta.audio_tokens is not None:
                # Process streaming audio tokens for real-time playback
                ...

        # For next generation, pass context:
        next_sample = ChatMLSample(
            messages=[...],
            misc={"audio_ids": generated_audio_ids}
        )
    """
    if stop_strings is None:
        stop_strings = ["<|end_of_text|>", "<|eot_id|>"]
    if ras_win_len is not None and ras_win_len <= 0:
        ras_win_len = None

    with torch.no_grad():
        inputs = self._prepare_inputs(chat_ml_sample, force_audio_gen=force_audio_gen)

        self._prepare_kv_caches()

        streamer = AsyncHiggsAudioStreamer(
            self.tokenizer,
            audio_num_codebooks=self.model.config.audio_num_codebooks,
            skip_prompt=True,
        )
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            stop_strings=stop_strings,
            tokenizer=self.tokenizer,
            do_sample=False if temperature == 0.0 else True,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            past_key_values_buckets=self.kv_caches,
            ras_win_len=ras_win_len,
            ras_win_max_num_repeat=ras_win_max_num_repeat,
            seed=seed,
            streamer=streamer,
        )
        thread = threading.Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # Accumulate audio tokens during streaming
        audio_token_frames = []

        async for delta in streamer:
            if delta.audio_tokens is not None:
                audio_token_frames.append(delta.audio_tokens.clone())
            yield delta

        # Wait for generation thread to complete
        thread.join()

        # Process accumulated tokens for context reuse (matching generation.py exactly)
        audio_context_tokens = None
        if audio_token_frames:
            # Stack frames: [num_codebooks] each -> [num_codebooks, num_frames]
            raw_audio_tokens = torch.stack(audio_token_frames, dim=1)

            # Apply delay pattern reversion if configured
            if self.model.config.use_delay_pattern:
                raw_audio_tokens = revert_delay_pattern(raw_audio_tokens)

            # Clip values and trim first/last tokens (matching generation.py: [:, 1:-1])
            audio_context_tokens = raw_audio_tokens.clip(0, self.audio_codebook_size - 1)[:, 1:-1]

        # Yield final delta with processed context tokens
        yield HiggsAudioStreamerDelta(
            finish_reason="stop",
            audio_context_tokens=audio_context_tokens,
        )
```

---

## Questions for Clarification

Before proceeding with implementation, I'd like to confirm:

1. **Rolling buffer size**: The example uses `generation_chunk_buffer_size = 2` (2 sentences). Should this be configurable via the API, or hardcoded?

2. **Message tracking**: generation.py also tracks `generation_messages` (user/assistant pairs). Should the caller manage this, or should we add helper methods?

3. **System message construction**: The `prepare_generation_context()` function in generation.py handles building the system message with `<|scene_desc_start|>` tags. Should we:
   - A) Document the expected format and let the caller build it
   - B) Add a helper function to serve_engine.py
   - C) Keep using the existing function from generation.py

4. **Thread safety**: The current implementation spawns a thread for generation. Is this acceptable, or should we use asyncio-native approaches?

---

## Summary of Changes

| File | Change | Lines Affected |
|------|--------|----------------|
| serve_engine.py | Add `audio_context_tokens` field to `HiggsAudioStreamerDelta` | ~30 |
| serve_engine.py | Modify `generate_delta_stream` to accumulate tokens, join thread, process, and yield final delta | 452-515 |

**No new classes or components required** - all changes extend existing functionality.
