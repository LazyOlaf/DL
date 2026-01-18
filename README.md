# DLChat

Real-time emotion-aware conversational assistant. Detects user emotion from speech (VAD: valence/arousal/dominance) and uses it to guide LLM responses.

## Quick Start

```bash
# Install system dependency
sudo apt install portaudio19-dev

# Create environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_phase_a.txt

# Download a GGUF model
mkdir -p models
# Place your model at models/mistral.gguf

# Run
python -m dlchat realtime              # With emotion
python -m dlchat realtime --no-emotion # Without emotion
python -m dlchat realtime --tts        # With TTS
python -m dlchat devices               # List audio devices
```

## How It Works

1. Microphone input with WebRTC VAD for utterance detection
2. faster-whisper transcribes speech
3. audeering model predicts VAD (valence/arousal/dominance)
4. Local LLM responds conditioned on transcript + VAD
5. Optional TTS speaks the response

The LLM receives VAD as numeric context:
```
[VAD: V=0.30, A=0.75, D=0.25 | prev: V=0.50, A=0.40, D=0.50]

I lost my job yesterday...
```

## Project Structure

```
dlchat/
  cli.py       # CLI entrypoint
  realtime.py  # Conversation loop
  affect.py    # Emotion detection
  asr.py       # Speech recognition
  audio.py     # Mic + VAD segmentation
  llm.py       # LLM wrapper
  prompts.py   # System prompts
  logging.py   # Turn logging
```

## Test Bench

`testbench_vad_conversations.py` tests emotion-aware vs plain responses across 9 scenarios.

## Logs

Each session logs to `runs/<timestamp>/`:
- `turns.jsonl`: Conversation with VAD, prompts, responses
- `audio/*.wav`: Utterance recordings
