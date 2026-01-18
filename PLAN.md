# DLChat - Real-time Emotion-Aware Conversational Agent

## Current Implementation

Real-time conversation with emotion detection from speech:
- **Audio capture** with WebRTC VAD for utterance segmentation
- **ASR** via faster-whisper
- **Emotion detection** via audeering wav2vec2 model (VAD: valence/arousal/dominance)
- **Local LLM** via llama-cpp-python
- **TTS** via pyttsx3 (optional)
- **Logging** to `runs/<timestamp>/`

## CLI Usage

```bash
python -m dlchat realtime              # With emotion
python -m dlchat realtime --no-emotion # Without emotion
python -m dlchat realtime --tts        # With TTS
python -m dlchat devices               # List audio devices
```

## Emotion-Aware Prompting

VAD values (0-1) are provided to the LLM as context:
```
[VAD: V=0.30, A=0.75, D=0.25 | prev: V=0.50, A=0.40, D=0.50]

User's message here...
```

The system prompt explains the VAD scale. The LLM interprets values naturally without explicit emotion labels.

## Project Structure

```
dlchat/
  cli.py       # CLI entrypoint
  realtime.py  # Main conversation loop
  affect.py    # Emotion detection (audeering VAD)
  asr.py       # Speech recognition (faster-whisper)
  audio.py     # Mic capture + utterance segmentation
  llm.py       # LLM wrapper (llama-cpp)
  prompts.py   # System prompts
  logging.py   # Turn logging
```

## Requirements

- Python 3.11+
- PortAudio (system): `sudo apt install portaudio19-dev`
- GGUF model file

## Test Bench

`testbench_vad_conversations.py` contains 9 two-turn scenarios for testing emotion-aware vs plain responses.

```bash
python testbench_vad_conversations.py              # All scenarios
python testbench_vad_conversations.py --scenario 3 # Single scenario
```
