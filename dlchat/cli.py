import argparse

from dlchat.app.realtime import run_realtime


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dlchat")
    sub = parser.add_subparsers(dest="cmd", required=True)

    dev = sub.add_parser("devices", help="List audio input/output devices")

    rt = sub.add_parser("realtime", help="Run Phase A realtime mic demo")
    rt.add_argument("--llm-gguf", default="models/mistral.gguf", help="Path to Mistral GGUF file")
    rt.add_argument("--asr-model", default="small.en", help="faster-whisper model size/name")
    rt.add_argument(
        "--affect-model-id",
        default="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
        help="HuggingFace model id for audio VAD regression",
    )
    rt.add_argument("--audio-device", type=int, default=None, help="sounddevice input device index")
    rt.add_argument("--vad-mode", type=int, default=2, choices=[0, 1, 2, 3], help="webrtcvad aggressiveness")
    rt.add_argument("--sample-rate", type=int, default=16000)
    rt.add_argument("--frame-ms", type=int, default=30, choices=[10, 20, 30])
    rt.add_argument("--silence-ms", type=int, default=900, help="end utterance after this much trailing silence")
    rt.add_argument("--min-utterance-ms", type=int, default=250, help="drop utterances shorter than this")
    rt.add_argument("--max-utterance-s", type=float, default=20.0, help="hard cap on utterance length")
    rt.add_argument("--n-ctx", type=int, default=16384, help="LLM context window (tokens)")
    rt.add_argument("--seed", type=int, default=42)
    rt.add_argument("--temperature", type=float, default=0.7)
    rt.add_argument("--top-p", type=float, default=0.9)
    rt.add_argument("--top-k", type=int, default=40)
    rt.add_argument("--repeat-penalty", type=float, default=1.10)
    rt.add_argument("--max-new-tokens", type=int, default=256)

    args = parser.parse_args(argv)

    if args.cmd == "devices":
        import sounddevice as sd

        print(sd.query_devices())
        return 0

    if args.cmd == "realtime":
        run_realtime(
            llm_gguf_path=args.llm_gguf,
            asr_model=args.asr_model,
            affect_model_id=args.affect_model_id,
            audio_device=args.audio_device,
            vad_mode=args.vad_mode,
            sample_rate=args.sample_rate,
            frame_ms=args.frame_ms,
            silence_ms=args.silence_ms,
            min_utterance_ms=args.min_utterance_ms,
            max_utterance_s=args.max_utterance_s,
            n_ctx=args.n_ctx,
            seed=args.seed,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repeat_penalty=args.repeat_penalty,
            max_new_tokens=args.max_new_tokens,
        )
        return 0

    raise RuntimeError("unreachable")
