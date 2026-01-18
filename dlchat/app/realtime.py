import json
import time

import numpy as np

from dlchat.asr.faster_whisper_asr import FasterWhisperAsr
from dlchat.audio.mic_stream import mic_frames
from dlchat.audio.utterance_vad import utterances_from_frames
from dlchat.llm.llama_cpp_model import LlamaCppModel
from dlchat.llm.mistral_instruct import build_mistral_instruct_prompt
from dlchat.logging.run_logger import RunLogger
from dlchat.logging.schema import AffectVAD, LLMDecoding, TurnRecord
from dlchat.prompts import EMOTION_AWARE_SYSTEM_PROMPT, NO_EMOTION_SYSTEM_PROMPT, format_emotion_context


def _pcm16_to_float32(pcm16: bytes) -> np.ndarray:
    audio_i16 = np.frombuffer(pcm16, dtype=np.int16)
    return (audio_i16.astype(np.float32) / 32768.0).copy()


def _ema(prev: AffectVAD | None, cur: AffectVAD, alpha: float = 0.30) -> AffectVAD:
    if prev is None:
        return cur
    return AffectVAD(
        valence=alpha * cur.valence + (1 - alpha) * prev.valence,
        arousal=alpha * cur.arousal + (1 - alpha) * prev.arousal,
        dominance=alpha * cur.dominance + (1 - alpha) * prev.dominance,
    )


def _init_tts():
    """Initialize TTS engine. Returns speak function or None if unavailable."""
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

    # Fallback: try espeak directly
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

    mode_str = "WITH emotion detection" if emotion_enabled else "WITHOUT emotion detection (plain mode)"
    print(f"\n[DLChat] Starting realtime conversation {mode_str}")
    if tts_enabled:
        print("[DLChat] TTS output enabled")
    print("[DLChat] Speak into your microphone. Press Ctrl+C to exit.\n")

    run_logger = RunLogger.create()
    asr = FasterWhisperAsr(model=asr_model)
    llm = LlamaCppModel(model_path=llm_gguf_path, n_ctx=n_ctx, seed=seed)

    # Only load affect model if emotion detection is enabled
    affect = None
    if emotion_enabled:
        from dlchat.affect.audeering_msp_dim import AudeeringMspDimRegressor
        from dlchat.affect.describe_vad import describe_vad
        affect = AudeeringMspDimRegressor(model_id=affect_model_id)

    # Initialize TTS if enabled
    speak_fn = None
    if tts_enabled:
        speak_fn = _init_tts()
        if speak_fn is None:
            print("[DLChat] Warning: TTS not available. Install pyttsx3 or espeak.")

    decoding = LLMDecoding(
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repeat_penalty=repeat_penalty,
        max_new_tokens=max_new_tokens,
    )

    history: list[tuple[str, str]] = []
    smooth: AffectVAD | None = None
    prev_vad: AffectVAD | None = None

    # Select system prompt based on mode
    system_prompt = EMOTION_AWARE_SYSTEM_PROMPT if emotion_enabled else NO_EMOTION_SYSTEM_PROMPT

    frames = mic_frames(sample_rate=sample_rate, frame_ms=frame_ms, device=audio_device)
    for turn_id, utt in enumerate(
        utterances_from_frames(
            frames,
            vad_mode=vad_mode,
            sample_rate=sample_rate,
            frame_ms=frame_ms,
            silence_ms=silence_ms,
            min_utterance_ms=min_utterance_ms,
            max_utterance_s=max_utterance_s,
        ),
        start=1,
    ):
        audio_path = run_logger.write_audio_wav(
            turn_id=turn_id,
            pcm16=utt.pcm16,
            sample_rate=sample_rate,
        )

        audio_f32 = _pcm16_to_float32(utt.pcm16)
        asr_text = asr.transcribe(audio_f32, sample_rate=sample_rate)

        # Process emotion if enabled
        vad_pred = None
        emotion_context = None
        state_json = None
        descriptors = []

        if emotion_enabled and affect is not None:
            from dlchat.affect.describe_vad import describe_vad

            vad_pred = affect.predict(audio_f32, sample_rate=sample_rate)
            smooth = _ema(smooth, vad_pred)
            descriptors = describe_vad(smooth)

            state_json = json.dumps(
                {
                    "valence": round(smooth.valence, 3),
                    "arousal": round(smooth.arousal, 3),
                    "dominance": round(smooth.dominance, 3),
                },
                ensure_ascii=False,
            )

            emotion_context = format_emotion_context(
                valence=smooth.valence,
                arousal=smooth.arousal,
                dominance=smooth.dominance,
                prev_valence=prev_vad.valence if prev_vad else None,
                prev_arousal=prev_vad.arousal if prev_vad else None,
                prev_dominance=prev_vad.dominance if prev_vad else None,
            )
            user_block = f"{emotion_context}\n\n{asr_text}"
            prev_vad = smooth
        else:
            user_block = asr_text

        prompt = build_mistral_instruct_prompt(
            system_prompt=system_prompt,
            history=history,
            user_message=user_block,
        )

        # Keep as much history as fits; drop oldest turns first.
        while llm.count_tokens(prompt) > (n_ctx - max_new_tokens - 256) and history:
            history = history[1:]
            prompt = build_mistral_instruct_prompt(
                system_prompt=system_prompt,
                history=history,
                user_message=user_block,
            )

        t0 = time.time()
        response = llm.complete(prompt, decoding=decoding)
        latency_s = time.time() - t0

        # Print output
        print("\n" + "=" * 80)
        print(f"Turn {turn_id} | latency={latency_s:.2f}s | audio={audio_path}")
        print(f"You: {asr_text}")
        if emotion_enabled and state_json:
            print(f"VAD: {state_json}")
        print(f"\nAssistant: {response}")

        # Speak response if TTS enabled
        if speak_fn is not None:
            speak_fn(response)

        # Log the turn
        record = TurnRecord(
            turn_id=turn_id,
            t_start=utt.t_start,
            t_end=utt.t_end,
            audio_wav_path=str(audio_path),
            asr_text=asr_text,
            affect_vad_raw=vad_pred if vad_pred else AffectVAD(0.5, 0.5, 0.5),
            affect_vad_smooth=smooth if smooth else AffectVAD(0.5, 0.5, 0.5),
            affect_descriptors=descriptors,
            llm_model_id="llm",
            llm_prompt=prompt,
            llm_decoding=decoding,
            llm_response_text=response,
        )
        run_logger.append_turn(record)

        history.append((user_block, response))
