# PLAN - Real-time Multimodal Conversational Agent with Mental-State Descriptors

## 0) Objective (aligned so far)
Build a human-like conversational chatbot that:
- takes live video + audio as input,
- predicts the user's mental state per utterance,
- responds based on both conversation context and predicted mental state,
- emphasizes "realism" primarily via conversational validation.

Research posture: prioritize correctness and minimal-but-understandable implementations that can be manually verified. Avoid production-hardening unless it improves correctness.

## 1) Confirmed constraints (from you)
- Start real-time from day 1.
- Fully local / open-source components (no external APIs).
- Domain: general conversation.
- Mental-state model uses strictly nonverbal signals (audio + video only; no transcript semantics).
- Time resolution: per utterance.
- Mental-state output: continuous dimensions.
- Use a suitable pretrained model if available; otherwise train our own.
- GPU: RTX 3090.
- Data policy: flexible.
- Dataset: IEMOCAP (kept at `IEMOCAP_full_release/`).
- Repo reset: delete everything except IEMOCAP (done; only `IEMOCAP_full_release/` + `PLAN.md` remain).

## 2) Mental state representation: dimensions -> descriptors
You asked for a list of mental state descriptors rather than a single emotion label, and you also prefer continuous dimensions. The cleanest research compromise is:
1) predict continuous dimensions (per utterance), then
2) derive a small descriptor list via an explicit, human-checkable mapping.

### 2.1 Default dimensions: VAD
Proposed dimensions: Valence, Arousal, Dominance (V, A, D).
Reasoning:
- VAD is standard, interpretable, and (importantly) IEMOCAP includes dimensional annotations we can train/evaluate against.
- A deterministic mapping from VAD to descriptors is auditable (no "mystery" labels).

### 2.2 Proposed descriptor vocabulary (derived from VAD)
Axis descriptors (always possible):
- Valence: `positive` vs `negative`
- Arousal: `energized` vs `calm`
- Dominance: `confident` vs `uncertain`

Composite descriptors (optional, still deterministic):
- `stressed` = negative + energized
- `down` = negative + calm
- `content` = positive + calm
- `excited` = positive + energized

Open choice: output only axis descriptors, or include composites too.
Confirmed: include axis descriptors + composites.

### 2.3 Mapping sketch (to finalize with you)
We will normalize VAD to a common internal scale **[0, 1]** and apply thresholds around 0.5:
- if |V - 0.5| <= t: include valence `neutral`; else include `negative` (< 0.5) or `positive` (> 0.5)
- if |A - 0.5| <= t: include arousal `neutral`; else include `calm` (< 0.5) or `energized` (> 0.5)
- if |D - 0.5| <= t: include dominance `neutral`; else include `uncertain` (< 0.5) or `confident` (> 0.5)
- if composites enabled: include `stressed/content/excited/down` based on (V, A) quadrants

Verbosity requirement (confirmed):
- Always emit 3 axis descriptors (valence/arousal/dominance), each as a short-but-elaborate phrase with intensity words where possible.
- Emit `neutral` per axis when |dim| is small (confirmed).
- Keep list length small (3 axis + optional 1 composite) so it can be used as an LLM conditioning signal without consuming too much context.

Reasoning: thresholding + quadrant rules are easy to inspect and tune, and avoid hidden prompt-based descriptor generation.

## 3) System architecture (real-time from day 1)
Two coupled pipelines run in real time: (A) perception and (B) dialogue.

### 3.1 Perception pipeline (nonverbal-only)
Goal: per utterance, output VAD + derived descriptors.

1) Capture
- Audio stream from microphone (Phase A affect baseline is audio-only)
- Video stream from webcam (optional early; not used for affect prediction in Phase A)

2) Utterance segmentation
- Use a local VAD (voice activity detector) to detect utterance boundaries on audio.
- Slice the corresponding video frame window by timestamps.

3) Feature extraction (baseline-first)
- Audio features (Phase A): pretrained speech encoder or minimal prosody.
- Video features (later): face-centered representation (face crop + CNN/ViT) or landmarks/AUs.

4) Fusion + regression head
- Phase A baseline: pooled audio embedding -> small MLP -> (V, A, D).
- Later: pooled audio + pooled video -> fusion -> (V, A, D).

5) Descriptor mapping
- Convert (V, A, D) -> descriptor list for the dialogue module.

### 3.2 Dialogue pipeline
Goal: generate a response that is coherent with context and appropriately validating given the predicted state.

1) ASR (local)
- Transcribe the user's utterance (for dialogue only; not used by the mental-state model).

2) Context manager
- Maintain transcript history; include as much as fits in the LLM context window, dropping oldest turns first (no summarization in the baseline).

3) Local LLM response
- Condition on:
  - conversation context,
  - predicted (V, A, D),
  - derived descriptor list,
  - the user's current utterance text.

Prompting constraint (confirmed):
- Do not add extra behavioral instructions (e.g., "be empathetic") beyond providing the mental-state signal and the user's query.
- Rely on each model's default instruction-tuning behavior; measure how much the *state signal alone* changes the response.

Decoding policy (confirmed):
- Fix decoding params across models for fair comparison.
- Fix seed so each model is internally reproducible given the same prompt.

System prompt (confirmed):
- Provide a short system prompt that defines what VAD in [0,1] means (0 = low, 0.5 = neutral, 1 = high, per axis) so the LLM can interpret the continuous values.

4) Output
- Start with text output; optionally add TTS later.

### 3.3 Logging (enabled by default; confirmed)
Log each utterance-turn with timestamps so experiments can be replayed and compared across models:
- raw audio segment for the utterance
- a small video clip (or sampled frames) covering the utterance window
- ASR transcript
- predicted VAD (raw + smoothed) and descriptor list
- the exact LLM prompt inputs + model id + response text

Reasoning: logging-by-default makes the project scientifically verifiable (we can audit alignment errors, compare LLMs fairly, and reproduce specific failures).

State visibility policy (confirmed):
- The state signal should condition the response, but we should not ask the assistant to output a rigid descriptor block.
- If the assistant naturally references affect ("you seem stressed"), that's acceptable; we simply avoid forcing a structured emotion report.

## 4) Research plan (phased, verifiable)
Phase A - Real-time skeleton (day 1 focus)
- Implement live capture + utterance segmentation + buffering/alignment.
- Plug in a placeholder mental-state model if needed, but ensure timing correctness and deterministic I/O.
- Integrate a local LLM for responses conditioned on predicted state.
Target platform (confirmed): Ubuntu 24.04 (CUDA-enabled) as the training/runtime reference environment.
Phase A affect baseline (confirmed): audio-only affect prediction.

Phase B - Offline training + evaluation (IEMOCAP)
- Build dataset loader: utterance -> audio (and later video) -> VAD target.
- Train a baseline audio-only VAD regressor first; add video after the real-time skeleton is stable.
- Report regression metrics + stability diagnostics; export a checkpoint for Phase A.

Phase C - Improve validation realism
- Keep prompt policy fixed (no extra behavioral instructions); improve realism via:
  - comparing Llama/Mistral/Qwen under identical prompts/decoding,
  - improving mental-state accuracy and descriptor mapping,
  - (optional later) local fine-tuning once baselines are stable.
- Evaluate with a small human rubric and scripted scenarios.

Phase D - Multimodal improvements
- Stronger encoders, better fusion, face tracking robustness, smoothing/hysteresis, etc.

## 5) Data: IEMOCAP usage plan
Dataset: `IEMOCAP_full_release/`

Intended use:
- Use per-utterance segmentation from IEMOCAP.
- Train on dimensional annotations (V, A, D) available in `Session*/dialog/EmoEvaluation/*.txt` headers like `[START - END] ... [V, A, D]`.
- Enforce nonverbal-only features during training/eval.
- Prefer session-based splits (e.g., leave-one-session-out) to reduce leakage.

## 6) Evaluation (high level)
Mental state (regression):
- MAE/RMSE per dimension; Pearson/Spearman correlation.
- Stability: jitter across adjacent utterances; optional smoothing impact.

Dialogue (validation realism):
- Human ratings: perceived understanding/validation, naturalness, appropriateness.
- Scripted tests: identical content with different predicted states -> check style controllability.

## 7) Alignment questions (resolved)
These are now resolved (answers recorded in section 9), kept here for traceability:

1) VAD target scale: keep IEMOCAP's native scale (likely 1-5), or normalize to [0, 1] internally?
2) Should we expose to the user: (a) only VAD, (b) only descriptor list, or (c) both?
3) Descriptor set: axis-only vs axis + composites (`stressed`, `content`, etc.)?
4) Real-time latency target: acceptable delay after speech ends (e.g., 300ms / 1s / 2s)?
5) Output modality: text-only, or text + TTS voice from the start?
6) ASR choice (local): Whisper / faster-whisper / other preference?
7) Local LLM choice: preferred family/license/size (e.g., Mistral 7B, Llama 3 8B, etc.)?
8) Validation style: explicit labeling ("you seem stressed") vs implicit validation ("that sounds like a lot")?
9) Smoothing: raw per-utterance VAD, or smoothed over last N utterances?
10) Data scope: use only IEMOCAP for mental-state training, or allow additional public datasets?

## 8) Decision log (append-only)
- 2026-01-18: Objective captured; initial plan drafted.
- 2026-01-18: Constraints confirmed (real-time, local OSS, nonverbal-only VAD per utterance; RTX 3090; IEMOCAP).
- 2026-01-18: Repo reset executed (deleted all files except `IEMOCAP_full_release/` and `PLAN.md`).
- 2026-01-18: Defaults chosen for alignment questions (VAD normalization; expose both; composites; ~1s latency; text-only; faster-whisper; implicit validation; mild smoothing; IEMOCAP-only; multi-LLM comparison).
- 2026-01-18: Phase A/B implementation constraints locked (voice-only; automatic VAD segmentation; logging enabled by default; llama.cpp runner; Linux/Ubuntu 24.04 CUDA training machine; no extra prompt instructions beyond state + query).
- 2026-01-18: LLM/ASR details locked (Llama/Mistral/Qwen checkpoints; GGUF `Q4_K_M`; faster-whisper `small.en`; emit `neutral` when |dim| is within threshold).
- 2026-01-18: Interface/measurement choices locked (max context history; fixed decoding params; state conditioning is internal/no rigid descriptor output; per-axis neutral + verbose descriptors; Phase A affect is audio-only).

## 9) Confirmed answers (update)
You chose the defaults for the open questions, with one extension for the local LLM:

1) VAD target scale: **normalize internally to [0, 1]**.
   - Reasoning: consistent descriptor mapping and a clean interface for real-time use; IEMOCAP targets can be mapped from 1-5 -> 0-1 for training/eval.
2) Expose outputs: **both** (VAD + descriptor list).
   - Reasoning: VAD is the ground-truth-aligned research target; the descriptor list is the product-facing conditioning signal for the chatbot.
3) Descriptor set: **axis + composites**.
   - Reasoning: composites (e.g., `stressed`) are more human-legible, while still being deterministic from VAD.
4) Real-time latency: **~1s** after end-of-speech.
   - Reasoning: gives enough time for ASR + mental-state inference without making the interaction feel sluggish.
5) Output modality: **text-only first**.
   - Reasoning: reduces moving parts early; keeps iteration tight for perception + dialogue coupling.
6) ASR: **faster-whisper** (local).
   - Reasoning: good accuracy/latency tradeoff; easy to run locally with GPU.
7) Local LLMs: **multiple models** (Llama, Mistral, plus a third such as Qwen).
   - Reasoning: we can treat LLM choice as an experimental factor for "validation realism", keeping the perception model fixed and swapping the dialogue model.
8) Validation prompting: **no additional behavioral instructions**.
   - Reasoning: this isolates the effect of the mental-state signal on each model's default instruction-following behavior.
9) Smoothing: **mild smoothing** (e.g., EMA or last-N=3).
   - Reasoning: reduces jitter in state conditioning without washing out quick affect changes.
10) Data scope: **start IEMOCAP-only**.
   - Reasoning: controls variables early; we can add extra datasets later if baseline is stable.
11) Exact LLM checkpoints (for fair ~8B-class comparison):
    - `Meta-Llama-3.1-8B-Instruct`
    - `Mistral-7B-Instruct-v0.3`
    - `Qwen2.5-7B-Instruct`
12) `llama.cpp` quantization: `Q4_K_M` for all three models.
13) faster-whisper model: `small.en`.
14) Descriptor mapping near zero: emit `neutral` when `|dim - 0.5| <= threshold`.
15) Conversation history: include as much history as fits in context window ("high N"), drop oldest first.
16) Decoding params: fixed across models (seed fixed for reproducibility).
17) State visibility: internal conditioning; assistant may mention affect naturally, but we do not request a structured descriptor output.
18) Descriptor style: per-axis-neutral descriptors should be verbose/elaborate (human-readable intensity + short explanations).
19) Phase A affect model: audio-only baseline.

## 10) Next alignment questions (to fully lock Phase A/B)
Resolved by your answers (kept here for traceability):
1) UI: CLI + optional OpenCV preview (default).
2) Conversation input: voice-only (mic).
3) Response output: text-only in Phase A (default).
4) LLM runner: `llama.cpp`/GGUF, and open-weight licensing (Llama) is acceptable.
5) Model sizes: fair comparison -> keep models in ~7B-8B class.
6) Validation behavior: do not add any extra instructions beyond the state signal and the user's query; use each LLM's default response.
7) Logging: enabled by default.
8) Utterance segmentation: automatic VAD segmentation (no push-to-talk).
9) Privacy/ethics: no privacy constraints; no (medical/diagnostic) claims.
10) Nonverbal-only mental state: yes (no ASR text into the VAD model).
11) Target platform: Linux; training machine is Ubuntu 24.04 with CUDA.

## 11) Remaining alignment questions (to start implementation cleanly)
Resolved (starting technical design now).

## 12) Technical architecture details (implementation-oriented)
This section turns the confirmed research choices into concrete interfaces/modules so the code stays small but coherent.

### 12.1 Core data model (turn-based)
We treat one "turn" as one user utterance detected by speech-VAD segmentation.

Proposed `TurnRecord` (logged as JSONL, one per utterance):
- `turn_id`: monotonically increasing int
- `t_start`, `t_end`: seconds (monotonic clock)
- `audio_wav_path`: saved utterance audio (WAV)
- `asr_text`: faster-whisper transcript
- `affect_vad_raw`: `{valence, arousal, dominance}` in [0, 1]
- `affect_vad_smooth`: same keys, smoothed
- `affect_descriptors`: list[str] (verbose; per-axis-neutral + composites)
- `llm_model_id`: one of {llama, mistral, qwen}
- `llm_prompt`: exact prompt text sent
- `llm_decoding`: fixed params (temperature/top_p/top_k/seed/etc.)
- `llm_response_text`: assistant response

### 12.2 Real-time pipeline scheduling (minimal, correct)
Single process, cooperating stages:
1) Audio capture -> rolling buffer
2) Speech-VAD segmentation -> emit utterance audio segments
3) For each utterance: ASR -> affect -> LLM -> log -> print response

We keep it intentionally simple (no distributed components) to maximize human verifiability.

### 12.3 Speech-VAD segmentation (not affect VAD)
Recommended baseline: a small, local VAD such as Silero VAD or WebRTC VAD.
Output: utterance boundaries with ~1s post-speech latency target.

### 12.4 Affect model baseline (audio-only first)
Training target: IEMOCAP dimensional annotations -> predict (V,A,D).
Baseline model idea (verifiable):
- pretrained audio encoder (frozen initially) -> mean pool -> small MLP -> 3 regressions.

Phase A (testable-from-day-1) choice:
- Use the pretrained audEERING dimensional emotion regressor `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim` (outputs arousal/dominance/valence ~ 0..1) as the initial audio-only affect model.

If a credible pretrained IEMOCAP VAD regressor exists and is compatible with our licensing constraints, we can swap it in later; otherwise train our own on IEMOCAP.

### 12.5 Descriptor mapping (verbose, per-axis-neutral)
Inputs: VAD in [0,1].
Outputs: list of short-but-elaborate phrases (always 3 axes + optional composite).

Neutral rule (confirmed): if |dim| <= threshold, emit a neutral descriptor for that axis (e.g., "valence: neutral").

### 12.6 LLM integration (3 models, llama.cpp)
Runner: `llama.cpp` via GGUF, quant `Q4_K_M` for all three models:
- `Meta-Llama-3.1-8B-Instruct`
- `Mistral-7B-Instruct-v0.3`
- `Qwen2.5-7B-Instruct`

Prompt structure (baseline):
- transcript history (as much as fits)
- a compact state block:
  - numeric (V,A,D)
  - descriptor list
- the user's current utterance

No extra instruction text beyond state + user query (confirmed).

### 12.7 Fixed decoding params (baseline; reproducible)
We will keep these identical across all 3 models for fair comparison and to make runs replayable:
- `seed`: 42
- `temperature`: 0.7
- `top_p`: 0.9
- `top_k`: 40
- `repeat_penalty`: 1.10
- `max_new_tokens`: 256

Reasoning:
- Non-zero temperature keeps conversation from becoming brittle, while a fixed seed keeps each model deterministic for a given prompt.
- A capped `max_new_tokens` bounds latency in real-time.

### 12.8 Context window + history policy (high N)
Baseline choices:
- Allocate a large context window (initial target: 16k tokens) to satisfy "as high N as possible".
- Maintain a transcript buffer; when near the context limit, drop oldest turns first (no summarization in baseline).
- Keep the state block compact (numbers + short descriptor list) so history dominates the token budget.

### 12.9 Descriptor mapping: concrete thresholds (baseline)
We need a small set of numeric constants so behavior is stable and human-checkable.

Baseline choices:
- VAD normalization from IEMOCAP 1-5 scale to [0, 1]: `(x - 1) / 4`
- Neutral half-band around 0.5: `t = 0.05` (i.e., neutral when `|x - 0.5| <= 0.05`)
- Intensity bins (used in wording):
  - `|x - 0.5| <= 0.05`: neutral
  - `0.05 < |x - 0.5| <= 0.15`: slight
  - `0.15 < |x - 0.5| <= 0.30`: moderate
  - `|x - 0.5| > 0.30`: strong

Descriptor phrasing guideline:
- Each axis descriptor is a single string, e.g. "valence: moderately negative (unpleasant/aversive)".
- Add at most one composite descriptor, e.g. "composite: stress-leaning (negative + high arousal)".

### 12.10 Logging layout (baseline)
Logging is enabled by default. One run creates a folder:
- `runs/<timestamp>/turns.jsonl`
- `runs/<timestamp>/audio/<turn_id>.wav`
- (optional later) `runs/<timestamp>/video/<turn_id>.mp4` or sampled frames

This makes it easy to replay the same turns through different LLMs and/or different affect models.

### 12.11 Minimal repository structure (proposed)
Keep the repo small and research-friendly:
- `dlchat/` (single Python package; runnable via `python -m dlchat ...`):
  - `app/` (orchestrator, CLI entrypoints)
  - `audio/` (mic capture + speech-VAD segmentation)
  - `asr/` (faster-whisper wrapper)
  - `affect/` (pretrained audio VAD regressor + descriptor mapping)
  - `llm/` (llama.cpp wrapper + prompt formatting)
  - `logging/` (TurnRecord + JSONL writer)
- `tests/` (lightweight unit tests)
- `runs/` (generated logs; not committed)
- `models/` (downloaded GGUF + whisper models; not committed)
- `tools/` (one-off scripts: replay, export, quick sanity checks; optional)

### 12.12 Phase A deliverable (definition of “working”)
Live demo is "working" when:
- speech is segmented into utterances reliably,
- each utterance yields a transcript (ASR),
- each utterance yields a VAD triple + descriptor list (placeholder OK at first),
- the LLM returns a response within the latency budget,
- every turn is logged (JSONL + WAV).

### 12.13 Phase B deliverable (IEMOCAP baseline model)
Offline training is "working" when:
- we can build a dataset index from `EmoEvaluation/*.txt` and locate each utterance WAV,
- we can train an audio-only regressor to predict (V,A,D) in [0,1],
- we can reproduce metrics from a saved checkpoint on a fixed split.

### 12.14 Replay protocol (fair LLM comparison)
Because real-time generation from 3 models at once is slow, the clean comparison path is:
1) run the live demo with 1 selected LLM and log all turns
2) replay the same logged turns offline through the other LLMs with identical prompts/decoding
3) compare responses + ratings side-by-side
