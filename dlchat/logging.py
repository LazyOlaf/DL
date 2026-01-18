"""Run logging for conversation turns."""
from __future__ import annotations

import json
import os
import wave
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from dlchat.affect import VAD
from dlchat.llm import LLMConfig


@dataclass(frozen=True)
class TurnRecord:
    turn_id: int
    t_start: float
    t_end: float
    audio_wav_path: str
    asr_text: str
    vad: VAD | None
    llm_prompt: str
    llm_config: LLMConfig
    llm_response: str

    def to_dict(self) -> dict:
        d = asdict(self)
        d["vad"] = asdict(self.vad) if self.vad else None
        d["llm_config"] = asdict(self.llm_config)
        return d


@dataclass(frozen=True)
class RunLogger:
    run_dir: Path
    audio_dir: Path
    turns_path: Path

    @staticmethod
    def create(root: str = "runs") -> "RunLogger":
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = Path(root) / ts
        audio_dir = run_dir / "audio"
        audio_dir.mkdir(parents=True, exist_ok=True)
        turns_path = run_dir / "turns.jsonl"
        turns_path.touch(exist_ok=True)
        return RunLogger(run_dir=run_dir, audio_dir=audio_dir, turns_path=turns_path)

    def write_audio_wav(self, *, turn_id: int, pcm16: bytes, sample_rate: int) -> Path:
        path = self.audio_dir / f"{turn_id:06d}.wav"
        with wave.open(str(path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16)
        return path

    def append_turn(self, turn: TurnRecord) -> None:
        line = json.dumps(turn.to_dict(), ensure_ascii=False)
        with self.turns_path.open("a", encoding="utf-8") as f:
            f.write(line + os.linesep)
