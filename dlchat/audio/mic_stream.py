from __future__ import annotations

import queue
import time
from dataclasses import dataclass
from typing import Iterator

import sounddevice as sd


@dataclass(frozen=True)
class MicFrame:
    pcm16: bytes
    t_start: float
    t_end: float


def mic_frames(*, sample_rate: int, frame_ms: int, device: int | None = None) -> Iterator[MicFrame]:
    frames_per_block = int(sample_rate * frame_ms / 1000)
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status):  # noqa: ANN001
        q.put(bytes(indata))

    with sd.RawInputStream(
        samplerate=sample_rate,
        blocksize=frames_per_block,
        dtype="int16",
        channels=1,
        device=device,
        callback=callback,
    ):
        t0 = time.monotonic()
        i = 0
        while True:
            pcm = q.get()
            t_start = (i * frames_per_block) / sample_rate
            t_end = ((i + 1) * frames_per_block) / sample_rate
            yield MicFrame(pcm16=pcm, t_start=t0 + t_start, t_end=t0 + t_end)
            i += 1
