"""
Higgs Audio Text Profile Module

Generates streaming audio using text description voice profiles for consistent
voice across multiple generations. Uses speaker_desc and scene_prompt in the
system message rather than voice clone (.wav/.txt) reference files.

Usage:
    from backend.higgs_audio_profile import (
        VoiceProfileSession,
        create_session,
        generate_audio_stream_with_profile,
    )

    # Create a session for consistent voice
    session = create_session(
        speaker_desc="Male, American accent, friendly tone, clear audio.",
        scene_prompt="Audio is recorded from a quiet room.",
    )

    # Generate audio for each text chunk
    async for pcm_bytes in generate_audio_stream_with_profile(
        engine=higgs_engine,
        text="Hello, how are you today?",
        session=session,
    ):
        # Stream pcm_bytes to client
        pass
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, AsyncGenerator

from backend.boson_multimodal.serve.serve_engine import HiggsAudioServeEngine
from backend.boson_multimodal.data_types import ChatMLSample, Message, AudioContent
from backend.boson_multimodal.model.higgs_audio.utils import revert_delay_pattern


@dataclass
class VoiceProfile:
    """Voice configuration using text description."""
    speaker_desc: str  # e.g., "Male, American accent, friendly tone, clear audio."
    scene_prompt: str  # e.g., "Audio is recorded from a quiet room."


@dataclass
class VoiceProfileSession:
    """
    Maintains state across generations for voice consistency.

    The key to consistent voice is passing previously generated audio_ids
    as context for subsequent generations. This session tracks:
    - voice_profile: The text description of the voice
    - generated_audio_ids: Audio tokens from previous generations
    - generation_messages: User/assistant message pairs for context
    - buffer_size: Max number of previous generations to retain
    """
    voice_profile: VoiceProfile
    generated_audio_ids: List[torch.Tensor] = field(default_factory=list)
    generation_messages: List[Message] = field(default_factory=list)
    buffer_size: int = 2  # Number of previous generations to keep for context


def create_session(
    speaker_desc: str,
    scene_prompt: str = "Audio is recorded from a quiet room.",
    buffer_size: int = 2,
) -> VoiceProfileSession:
    """
    Create a new voice profile session.

    Args:
        speaker_desc: Text description of the voice characteristics.
            Example: "Male, American accent, modern speaking rate, moderate-pitch,
                     friendly tone, and very clear audio."
        scene_prompt: Description of the audio recording environment.
            Example: "Audio is recorded from a quiet room."
        buffer_size: Number of previous generations to retain for context.
            Higher values = more consistent voice but more memory/compute.

    Returns:
        VoiceProfileSession ready for audio generation.
    """
    voice_profile = VoiceProfile(
        speaker_desc=speaker_desc,
        scene_prompt=scene_prompt,
    )
    return VoiceProfileSession(
        voice_profile=voice_profile,
        generated_audio_ids=[],
        generation_messages=[],
        buffer_size=buffer_size,
    )


def reset_session(session: VoiceProfileSession) -> None:
    """
    Clear accumulated context while keeping voice profile settings.

    Call this to start fresh generations without changing the voice.
    """
    session.generated_audio_ids.clear()
    session.generation_messages.clear()


def build_system_message(voice_profile: VoiceProfile) -> Message:
    """
    Build system message with speaker description and scene prompt.

    Format follows generation.py pattern with <|scene_desc_start|> tags.
    The model uses this to understand the desired voice characteristics.
    """
    content_parts = ["Generate audio following instruction."]

    scene_desc_parts = []
    if voice_profile.scene_prompt:
        scene_desc_parts.append(voice_profile.scene_prompt)

    # Add speaker description (single speaker = SPEAKER0)
    scene_desc_parts.append(f"SPEAKER0: {voice_profile.speaker_desc}")

    scene_desc = "\n\n".join(scene_desc_parts)
    content_parts.append(f"<|scene_desc_start|>\n{scene_desc}\n<|scene_desc_end|>")

    return Message(
        role="system",
        content="\n\n".join(content_parts),
    )


def _prepare_context_audio_ids(
    session: VoiceProfileSession,
) -> Optional[List[torch.Tensor]]:
    """
    Prepare the audio_ids list for context.

    Returns None if no previous generations exist,
    otherwise returns the list of audio tensors.
    """
    if not session.generated_audio_ids:
        return None
    return session.generated_audio_ids


def _update_session_after_generation(
    session: VoiceProfileSession,
    text: str,
    audio_ids: torch.Tensor,
) -> None:
    """
    Update session state after a successful generation.

    Adds the new audio_ids and messages to the session,
    then applies buffer size limit to prevent unbounded growth.
    """
    # Add the generated audio tokens
    session.generated_audio_ids.append(audio_ids)

    # Add the user message (the text that was spoken)
    session.generation_messages.append(
        Message(role="user", content=text)
    )

    # Add assistant message (AudioContent placeholder for the generated audio)
    session.generation_messages.append(
        Message(role="assistant", content=AudioContent(audio_url=""))
    )

    # Apply buffer limit (from generation.py:368-370)
    if session.buffer_size and len(session.generated_audio_ids) > session.buffer_size:
        session.generated_audio_ids = session.generated_audio_ids[-session.buffer_size:]
        # Messages are pairs (user + assistant), so multiply by 2
        session.generation_messages = session.generation_messages[(-2 * session.buffer_size):]


async def generate_audio_stream_with_profile(
    engine: HiggsAudioServeEngine,
    text: str,
    session: VoiceProfileSession,
    max_new_tokens: int = 2048,
    temperature: float = 0.7,
    top_p: float = 0.95,
    top_k: int = 50,
    ras_win_len: int = 7,
    ras_win_max_num_repeat: int = 2,
    chunk_size: int = 14,
) -> AsyncGenerator[bytes, None]:
    """
    Generate streaming audio using text profile method.

    This function maintains voice consistency by:
    1. Using speaker_desc in system message to define voice characteristics
    2. Passing previously generated audio_ids as context
    3. Including previous messages in the conversation for continuity

    Args:
        engine: Initialized HiggsAudioServeEngine instance.
        text: The text to synthesize into speech.
        session: VoiceProfileSession with voice config and accumulated context.
        max_new_tokens: Maximum audio tokens to generate.
        temperature: Sampling temperature (higher = more variation).
        top_p: Nucleus sampling threshold.
        top_k: Top-k sampling parameter.
        ras_win_len: RAS (Repetition Aware Sampling) window length.
        ras_win_max_num_repeat: Max repetitions allowed in RAS window.
        chunk_size: Number of tokens to accumulate before decoding a chunk.

    Yields:
        PCM16 audio bytes (24kHz, mono, 16-bit signed).
    """
    # Build the full message list
    messages = []

    # 1. System message with voice profile
    system_message = build_system_message(session.voice_profile)
    messages.append(system_message)

    # 2. Previous generation messages (user text + assistant audio pairs)
    messages.extend(session.generation_messages)

    # 3. Current user message (text to synthesize)
    messages.append(Message(role="user", content=text))

    # Prepare context audio_ids
    context_audio_ids = _prepare_context_audio_ids(session)

    # Create ChatMLSample with audio context in misc
    misc = None
    if context_audio_ids:
        misc = {"audio_ids": context_audio_ids}

    chat_sample = ChatMLSample(
        messages=messages,
        misc=misc,
    )

    # Streaming state
    audio_tokens: List[torch.Tensor] = []
    seq_len = 0
    device = engine.device

    # Stream generation
    async for delta in engine.generate_delta_stream(
        chat_ml_sample=chat_sample,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        stop_strings=["<|end_of_text|>", "<|eot_id|>"],
        ras_win_len=ras_win_len,
        ras_win_max_num_repeat=ras_win_max_num_repeat,
        force_audio_gen=True,
    ):
        if delta.audio_tokens is None:
            continue

        # Check for end token (1025)
        if torch.all(delta.audio_tokens == 1025):
            break

        # Accumulate tokens
        audio_tokens.append(delta.audio_tokens[:, None])

        # Count non-padding tokens (1024 is padding)
        if torch.all(delta.audio_tokens != 1024):
            seq_len += 1

        # Decode when chunk size reached
        if seq_len > 0 and seq_len % chunk_size == 0:
            audio_tensor = torch.cat(audio_tokens, dim=-1)

            try:
                # Revert delay pattern and decode
                vq_code = (
                    revert_delay_pattern(audio_tensor, start_idx=seq_len - chunk_size + 1)
                    .clip(0, 1023)
                    .to(device)
                )
                waveform = engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

                # Convert to numpy
                if isinstance(waveform, torch.Tensor):
                    waveform_np = waveform.detach().cpu().numpy()
                else:
                    waveform_np = np.asarray(waveform, dtype=np.float32)

                # Convert to PCM16 bytes
                pcm = np.clip(waveform_np, -1.0, 1.0)
                pcm16 = (pcm * 32767.0).astype(np.int16)
                yield pcm16.tobytes()

            except Exception as e:
                # Log but continue - don't break streaming on decode errors
                continue

    # Flush remaining tokens
    if seq_len > 0 and seq_len % chunk_size != 0 and audio_tokens:
        audio_tensor = torch.cat(audio_tokens, dim=-1)
        remaining = seq_len % chunk_size

        try:
            vq_code = (
                revert_delay_pattern(audio_tensor, start_idx=seq_len - remaining + 1)
                .clip(0, 1023)
                .to(device)
            )
            waveform = engine.audio_tokenizer.decode(vq_code.unsqueeze(0))[0, 0]

            if isinstance(waveform, torch.Tensor):
                waveform_np = waveform.detach().cpu().numpy()
            else:
                waveform_np = np.asarray(waveform, dtype=np.float32)

            pcm = np.clip(waveform_np, -1.0, 1.0)
            pcm16 = (pcm * 32767.0).astype(np.int16)
            yield pcm16.tobytes()

        except Exception:
            pass

    # Update session with this generation's audio for future context
    if audio_tokens:
        final_audio_tensor = torch.cat(audio_tokens, dim=-1)
        # Store the raw audio tokens (before delay pattern reversion)
        # The serve_engine._prepare_inputs expects tokens in original format
        _update_session_after_generation(session, text, final_audio_tensor)


# Convenience function matching fastserver.py Speech interface
async def generate_audio_for_sentence(
    engine: HiggsAudioServeEngine,
    text: str,
    session: VoiceProfileSession,
    chunk_size: int = 14,
) -> AsyncGenerator[bytes, None]:
    """
    Convenience wrapper matching the fastserver.py Speech.generate_audio_for_sentence interface.

    Args:
        engine: Initialized HiggsAudioServeEngine instance.
        text: The sentence text to synthesize.
        session: VoiceProfileSession for voice consistency.
        chunk_size: Tokens per decoded chunk.

    Yields:
        PCM16 audio bytes.
    """
    async for pcm_bytes in generate_audio_stream_with_profile(
        engine=engine,
        text=text,
        session=session,
        chunk_size=chunk_size,
    ):
        yield pcm_bytes
