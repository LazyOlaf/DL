# DLChat

Real-time emotion-aware conversational assistant. Detects user emotion from speech (VAD: valence/arousal/dominance) and uses it to guide LLM responses.

## Quick Start

```bash
# Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_phase_a.txt

# Download a GGUF model (example: TinyLlama for testing)
mkdir -p models
# Place your GGUF model at models/mistral.gguf

# Run with emotion detection
python -m dlchat realtime

# Run without emotion detection (for comparison)
python -m dlchat realtime --no-emotion

# Run with TTS output
python -m dlchat realtime --tts
```

## CLI Options

```
python -m dlchat realtime [OPTIONS]

Options:
  --llm-gguf PATH         Path to GGUF model (default: models/mistral.gguf)
  --asr-model NAME        Whisper model size (default: small.en)
  --audio-device INDEX    Microphone device index
  --no-emotion            Disable emotion detection (plain LLM mode)
  --tts                   Enable text-to-speech output
  --n-ctx INT             LLM context window (default: 16384)
  --temperature FLOAT     LLM temperature (default: 0.7)

python -m dlchat devices   # List audio devices
```

## How It Works

1. **Audio capture**: Microphone input with WebRTC VAD for utterance detection
2. **ASR**: faster-whisper transcribes speech to text
3. **Emotion detection**: audeering model predicts VAD (valence/arousal/dominance)
4. **LLM response**: Local LLM generates response conditioned on transcript + VAD
5. **TTS** (optional): pyttsx3 speaks the response

### Emotion-Aware Prompting

The LLM receives VAD as numeric context:
```
[VAD: V=0.30, A=0.75, D=0.25 | prev: V=0.50, A=0.40, D=0.50]

I lost my job yesterday...
```

The system prompt explains the VAD scale. The LLM interprets changes naturally without explicit emotion labels.

## Requirements

- Python 3.11+
- CUDA (optional, for GPU acceleration)
- Microphone
- GGUF model file

## Project Structure

```
dlchat/
  app/realtime.py    # Main conversation loop
  asr/               # Speech recognition (faster-whisper)
  affect/            # Emotion detection (audeering VAD model)
  audio/             # Mic capture and utterance segmentation
  llm/               # LLM wrapper (llama-cpp-python)
  prompts.py         # System prompts for emotion-aware mode
  logging/           # Turn logging (JSONL + WAV)
```

## Logs

Each session logs to `runs/<timestamp>/`:
- `turns.jsonl`: Full conversation with VAD, prompts, responses
- `audio/*.wav`: Individual utterance recordings
