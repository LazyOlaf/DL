"""Audio capture and utterance segmentation."""
from __future__ import annotations

import queue
import time
from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator

import sounddevice as sd
import webrtcvad


@dataclass(frozen=True)
class MicFrame:
    pcm16: bytes
    t_start: float
    t_end: float


@dataclass(frozen=True)
class Utterance:
    pcm16: bytes
    t_start: float
    t_end: float


def mic_frames(*, sample_rate: int, frame_ms: int, device: int | None = None) -> Iterator[MicFrame]:
    """Yield audio frames from microphone."""
    frames_per_block = int(sample_rate * frame_ms / 1000)
    q: queue.Queue[bytes] = queue.Queue()

    def callback(indata, frames, time_info, status):
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


def segment_utterances(
    frames: Iterable[MicFrame],
    *,
    vad_mode: int,
    sample_rate: int,
    frame_ms: int,
    silence_ms: int,
    min_utterance_ms: int,
    max_utterance_s: float,
) -> Iterator[Utterance]:
    """Segment audio frames into utterances using WebRTC VAD."""
    vad = webrtcvad.Vad(vad_mode)
    padding_frames = max(1, int(silence_ms / frame_ms))
    ring: deque[tuple[MicFrame, bool]] = deque(maxlen=padding_frames)

    triggered = False
    voiced: list[MicFrame] = []

    for frame in frames:
        is_speech = vad.is_speech(frame.pcm16, sample_rate)

        if not triggered:
            ring.append((frame, is_speech))
            if len(ring) == ring.maxlen and sum(1 for _, s in ring if s) > 0.8 * ring.maxlen:
                triggered = True
                voiced.extend([f for f, _ in ring])
                ring.clear()
            continue

        voiced.append(frame)
        ring.append((frame, is_speech))

        trailing_silence = len(ring) == ring.maxlen and sum(1 for _, s in ring if not s) > 0.8 * ring.maxlen
        duration_s = voiced[-1].t_end - voiced[0].t_start

        if trailing_silence or duration_s >= max_utterance_s:
            pcm = b"".join(f.pcm16 for f in voiced)
            dur_ms = int(1000 * len(pcm) / 2 / sample_rate)
            if dur_ms >= min_utterance_ms:
                yield Utterance(pcm16=pcm, t_start=voiced[0].t_start, t_end=voiced[-1].t_end)

            triggered = False
            ring.clear()
            voiced = []
