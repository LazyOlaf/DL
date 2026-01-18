"""Real-time conversation loop."""
from __future__ import annotations

import time

import numpy as np

from dlchat.affect import AffectModel, VAD
from dlchat.asr import ASR
from dlchat.audio import mic_frames, segment_utterances
from dlchat.llm import LLM, LLMConfig, build_prompt
from dlchat.logging import RunLogger, TurnRecord
from dlchat.prompts import EMOTION_SYSTEM_PROMPT, PLAIN_SYSTEM_PROMPT, format_vad_context


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0).copy()


def _ema(prev: VAD | None, cur: VAD, alpha: float = 0.30) -> VAD:
    if prev is None:
        return cur
    return VAD(
        valence=alpha * cur.valence + (1 - alpha) * prev.valence,
        arousal=alpha * cur.arousal + (1 - alpha) * prev.arousal,
        dominance=alpha * cur.dominance + (1 - alpha) * prev.dominance,
    )


def _init_tts():
    """Initialize TTS engine. Returns speak function or None."""
    try:
        import pyttsx3
        engine = pyttsx3.init()
        engine.setProperty('rate', 175)
        def speak(text: str) -> None:
            engine.say(text)
            engine.runAndWait()
        return speak
    except Exception:
        pass
    try:
        import subprocess
        result = subprocess.run(["espeak", "--version"], capture_output=True)
        if result.returncode == 0:
            def speak(text: str) -> None:
                subprocess.run(["espeak", text], capture_output=True)
            return speak
    except Exception:
        pass
    return None


def run_realtime(
    *,
    llm_gguf_path: str,
    asr_model: str,
    affect_model_id: str,
    audio_device: int | None,
    vad_mode: int,
    sample_rate: int,
    frame_ms: int,
    silence_ms: int,
    min_utterance_ms: int,
    max_utterance_s: float,
    n_ctx: int,
    seed: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repeat_penalty: float,
    max_new_tokens: int,
    emotion_enabled: bool = True,
    tts_enabled: bool = False,
) -> None:
    """Run the realtime conversation loop."""

    mode_str = "WITH emotion" if emotion_enabled else "WITHOUT emotion"
    print(f"\n[DLChat] Starting realtime conversation {mode_str}")
    if tts_enabled:
        print("[DLChat] TTS enabled")
    print("[DLChat] Speak into your microphone. Ctrl+C to exit.\n")

    logger = RunLogger.create()
    asr = ASR(model=asr_model)
    llm = LLM(model_path=llm_gguf_path, n_ctx=n_ctx, seed=seed)

    affect = None
    if emotion_enabled:
        affect = AffectModel(model_id=affect_model_id)

    speak_fn = _init_tts() if tts_enabled else None
    if tts_enabled and speak_fn is None:
        print("[DLChat] Warning: TTS not available")

    config = LLMConfig(
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        max_new_tokens=max_new_tokens,
    )

    history: list[tuple[str, str]] = []
    smooth_vad: VAD | None = None
    prev_vad: VAD | None = None
    system_prompt = EMOTION_SYSTEM_PROMPT if emotion_enabled else PLAIN_SYSTEM_PROMPT

    frames = mic_frames(sample_rate=sample_rate, frame_ms=frame_ms, device=audio_device)
    utterances = segment_utterances(
        frames,
        vad_mode=vad_mode,
        sample_rate=sample_rate,
        frame_ms=frame_ms,
        silence_ms=silence_ms,
        min_utterance_ms=min_utterance_ms,
        max_utterance_s=max_utterance_s,
    )

    for turn_id, utt in enumerate(utterances, start=1):
        audio_path = logger.write_audio_wav(turn_id=turn_id, pcm16=utt.pcm16, sample_rate=sample_rate)
        audio_f32 = _pcm16_to_float32(utt.pcm16)
        asr_text = asr.transcribe(audio_f32, sample_rate=sample_rate)

        vad_pred = None
        if emotion_enabled and affect is not None:
            vad_pred = affect.predict(audio_f32, sample_rate=sample_rate)
            smooth_vad = _ema(smooth_vad, vad_pred)
            vad_context = format_vad_context(
                smooth_vad.valence, smooth_vad.arousal, smooth_vad.dominance,
                prev_vad.valence if prev_vad else None,
                prev_vad.arousal if prev_vad else None,
                prev_vad.dominance if prev_vad else None,
            )
            user_block = f"{vad_context}\n\n{asr_text}"
            prev_vad = smooth_vad
        else:
            user_block = asr_text

        prompt = build_prompt(system_prompt=system_prompt, history=history, user_message=user_block)

        # Trim history if needed
        while llm.count_tokens(prompt) > (n_ctx - max_new_tokens - 256) and history:
            history = history[1:]
            prompt = build_prompt(system_prompt=system_prompt, history=history, user_message=user_block)

        t0 = time.time()
        response = llm.complete(prompt, config=config)
        latency_s = time.time() - t0

        print("\n" + "=" * 60)
        print(f"Turn {turn_id} | {latency_s:.2f}s | {audio_path}")
        print(f"You: {asr_text}")
        if smooth_vad:
            print(f"VAD: V={smooth_vad.valence:.2f} A={smooth_vad.arousal:.2f} D={smooth_vad.dominance:.2f}")
        print(f"\nAssistant: {response}")

        if speak_fn:
            speak_fn(response)

        record = TurnRecord(
            turn_id=turn_id,
            t_start=utt.t_start,
            t_end=utt.t_end,
            audio_wav_path=str(audio_path),
            asr_text=asr_text,
            vad=smooth_vad,
            llm_prompt=prompt,
            llm_config=config,
            llm_response=response,
        )
        logger.append_turn(record)
        history.append((user_block, response))
