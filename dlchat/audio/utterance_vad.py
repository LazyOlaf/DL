from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Iterator

import webrtcvad

from dlchat.audio.mic_stream import MicFrame


@dataclass(frozen=True)
class Utterance:
    pcm16: bytes
    t_start: float
    t_end: float


def utterances_from_frames(
    frames: Iterable[MicFrame],
    *,
    vad_mode: int,
    sample_rate: int,
    frame_ms: int,
    silence_ms: int,
    min_utterance_ms: int,
    max_utterance_s: float,
) -> Iterator[Utterance]:
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
                yield Utterance(
                    pcm16=pcm,
                    t_start=voiced[0].t_start,
                    t_end=voiced[-1].t_end,
                )

            triggered = False
            ring.clear()
            voiced = []
