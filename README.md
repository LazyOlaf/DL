# DLChat (Phase A)

Real-time (mic -> speech segmentation -> ASR -> affect VAD -> local LLM response), research-first and fully local.

## Phase A baseline (what's implemented)
- **Audio-only affect** via pretrained audEERING dimensional emotion regressor:
  - Model: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
  - Outputs: continuous `valence/arousal/dominance` ~ `[0, 1]`
- **ASR** via `faster-whisper` (`small.en`).
- **LLM** via `llama-cpp-python` + a local GGUF (default target: `Mistral-7B-Instruct-v0.3`).
- **Logging enabled by default** to `runs/<timestamp>/` (JSONL + utterance WAVs).

## Quickstart (Ubuntu 24.04 recommended)
Prereqs:
- Python 3.11
- Build tools for `llama-cpp-python` (e.g., `build-essential`, `cmake`)
- Audio I/O (e.g., `portaudio19-dev` for `sounddevice`)

GPU notes (optional, for speed):
- `llama-cpp-python` needs to be built with CUDA for GPU offload (see its docs; typically `CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install --no-binary llama-cpp-python llama-cpp-python`).
- `faster-whisper` uses `ctranslate2`; install a CUDA-enabled build if you want GPU ASR.

Install:
1. `python3.11 -m venv .venv`
2. `. .venv/bin/activate`
3. `pip install -r requirements_phase_a.txt`

Models:
- Put your Mistral GGUF at `models/mistral.gguf` (or pass `--llm-gguf`).
  - Recommended quant (fair baseline): `Q4_K_M`
- First run will auto-download:
  - `faster-whisper` model files
  - the audEERING affect model from Hugging Face

Run (mic):
- `python -m dlchat realtime` (expects `models/mistral.gguf`)
  - If you need to select a microphone: `python -m dlchat devices` then re-run with `--audio-device <index>`

Output:
- Prints transcript, predicted VAD JSON (`[0,1]` floats), and the LLM response.
- Logs everything under `runs/<timestamp>/`.

## Notes
- The LLM gets a **system prompt** that defines what each `[0,1]` axis means (0 = low, 0.5 = neutral, 1 = high).
- We do **not** instruct response style; we only provide state + the user's utterance text.
