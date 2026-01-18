from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class AffectVAD:
    valence: float
    arousal: float
    dominance: float


@dataclass(frozen=True)
class LLMDecoding:
    seed: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    max_new_tokens: int


@dataclass(frozen=True)
class TurnRecord:
    turn_id: int
    t_start: float
    t_end: float
    audio_wav_path: str
    asr_text: str
    affect_vad_raw: AffectVAD
    affect_vad_smooth: AffectVAD
    affect_descriptors: list[str]
    llm_model_id: str
    llm_prompt: str
    llm_decoding: LLMDecoding
    llm_response_text: str

    def to_dict(self) -> dict:
        return asdict(self)
