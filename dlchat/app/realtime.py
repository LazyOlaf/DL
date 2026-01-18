import json
import time

import numpy as np

from dlchat.affect.audeering_msp_dim import AudeeringMspDimRegressor
from dlchat.affect.describe_vad import describe_vad
from dlchat.asr.faster_whisper_asr import FasterWhisperAsr
from dlchat.audio.mic_stream import mic_frames
from dlchat.audio.utterance_vad import utterances_from_frames
from dlchat.llm.llama_cpp_model import LlamaCppModel
from dlchat.llm.mistral_instruct import build_mistral_instruct_prompt
from dlchat.logging.run_logger import RunLogger
from dlchat.logging.schema import AffectVAD, LLMDecoding, TurnRecord

SYSTEM_PROMPT_VAD_01 = """You are a conversational assistant.

You are given an estimated mental-state signal for the current user utterance as three continuous scalars in [0, 1]:

- valence: 0 = very negative/unpleasant, 0.5 = neutral, 1 = very positive/pleasant
- arousal: 0 = very calm/low activation, 0.5 = neutral, 1 = very activated/energetic
- dominance: 0 = low dominance (uncertain/out of control), 0.5 = neutral, 1 = high dominance (confident/in control)

This signal is noisy and should be treated as soft context.
Do not output a rigid emotion report unless the user asks; just respond naturally to what the user said.
"""


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
) -> None:
    run_logger = RunLogger.create()
    asr = FasterWhisperAsr(model=asr_model)
    affect = AudeeringMspDimRegressor(model_id=affect_model_id)
    llm = LlamaCppModel(model_path=llm_gguf_path, n_ctx=n_ctx, seed=seed)

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

        vad_pred = affect.predict(audio_f32, sample_rate=sample_rate)
        smooth = _ema(smooth, vad_pred)
        descriptors = describe_vad(smooth)

        state_json = json.dumps(
            {
                "valence": smooth.valence,
                "arousal": smooth.arousal,
                "dominance": smooth.dominance,
            },
            ensure_ascii=False,
        )

        user_block = (
            "MENTAL_STATE_JSON:\n"
            f"{state_json}\n\n"
            "MENTAL_STATE_DESCRIPTORS:\n"
            + "\n".join(f"- {d}" for d in descriptors)
            + "\n\n"
            f"USER:\n{asr_text}\n"
        )

        prompt = build_mistral_instruct_prompt(
            system_prompt=SYSTEM_PROMPT_VAD_01,
            history=history,
            user_message=user_block,
        )

        # Keep as much history as fits; drop oldest turns first.
        while llm.count_tokens(prompt) > (n_ctx - max_new_tokens - 256) and history:
            history = history[1:]
            prompt = build_mistral_instruct_prompt(
                system_prompt=SYSTEM_PROMPT_VAD_01,
                history=history,
                user_message=user_block,
            )

        t0 = time.time()
        response = llm.complete(
            prompt,
            decoding=decoding,
        )
        latency_s = time.time() - t0

        print("\n" + "=" * 80)
        print(f"Turn {turn_id} | latency={latency_s:.2f}s | audio={audio_path}")
        print(f"ASR: {asr_text}")
        print(f"VAD: {state_json}")
        print("Descriptors:")
        for d in descriptors:
            print(f"  - {d}")
        print("\nAssistant:")
        print(response)

        record = TurnRecord(
            turn_id=turn_id,
            t_start=utt.t_start,
            t_end=utt.t_end,
            audio_wav_path=str(audio_path),
            asr_text=asr_text,
            affect_vad_raw=vad_pred,
            affect_vad_smooth=smooth,
            affect_descriptors=descriptors,
            llm_model_id="mistral",
            llm_prompt=prompt,
            llm_decoding=decoding,
            llm_response_text=response,
        )
        run_logger.append_turn(record)

        history.append((user_block, response))
