#!/usr/bin/env python3
"""
Test bench for emotion-aware vs plain LLM responses.
9 two-turn conversations where VAD shifts demonstrate emotion-aware value.
"""
import argparse
from dataclasses import dataclass

from llama_cpp import Llama

from dlchat.llm import build_tinyllama_prompt

# Simple prompts for small models
EMOTION_SYSTEM = "You are a helpful assistant. The user's mood is shown in brackets. Respond with empathy."
PLAIN_SYSTEM = "You are a helpful assistant."


def vad_to_mood(v: float, a: float, d: float) -> str:
    """Convert VAD to mood word."""
    if v < 0.3:
        return "distressed" if a > 0.6 and d < 0.4 else "frustrated" if a > 0.6 else "sad" if d < 0.4 else "disappointed"
    elif v > 0.7:
        return "excited" if a > 0.6 and d > 0.5 else "nervous" if a > 0.6 else "content" if d > 0.5 else "relieved"
    else:
        return "anxious" if a > 0.6 and d < 0.4 else "tense" if a > 0.6 else "calm" if d > 0.5 else "uncertain"


@dataclass
class Turn:
    text: str
    valence: float
    arousal: float
    dominance: float

    @property
    def mood(self) -> str:
        return vad_to_mood(self.valence, self.arousal, self.dominance)


@dataclass
class Scenario:
    name: str
    description: str
    turn1: Turn
    turn2: Turn


SCENARIOS = [
    Scenario("job_loss_distress", "User shares job loss, becomes more distressed",
             Turn("I just got laid off from my job today.", 0.25, 0.55, 0.30),
             Turn("I don't know what I'm going to do.", 0.15, 0.70, 0.15)),
    Scenario("excitement_dampened", "User excited, enthusiasm drops",
             Turn("I just got accepted into my dream university!", 0.90, 0.85, 0.80),
             Turn("Yeah I guess it is a big deal.", 0.60, 0.40, 0.55)),
    Scenario("anxiety_escalation", "User anxious, anxiety increases",
             Turn("I have a big presentation tomorrow and I'm nervous.", 0.35, 0.70, 0.35),
             Turn("What if I completely freeze up in front of everyone?", 0.20, 0.85, 0.20)),
    Scenario("grief_deepening", "User mentions loss, grief intensifies",
             Turn("My grandmother passed away last week.", 0.20, 0.40, 0.30),
             Turn("I keep expecting to hear her voice when I call home.", 0.10, 0.50, 0.20)),
    Scenario("frustration_building", "User frustrated, frustration builds",
             Turn("My roommate keeps leaving dishes in the sink.", 0.35, 0.60, 0.50),
             Turn("I've asked them five times already and nothing changes!", 0.25, 0.80, 0.40)),
    Scenario("hope_fading", "User hopeful, hope fades",
             Turn("I think my ex might want to get back together.", 0.65, 0.60, 0.50),
             Turn("Actually they just texted saying they're seeing someone new.", 0.20, 0.55, 0.25)),
    Scenario("overwhelmed_parent", "Parent struggling, more overwhelmed",
             Turn("Balancing work and taking care of the kids is hard.", 0.40, 0.55, 0.35),
             Turn("I feel like I'm failing at both.", 0.25, 0.65, 0.20)),
    Scenario("health_anxiety", "User worried about health, worry intensifies",
             Turn("I found a lump and I'm waiting for test results.", 0.30, 0.65, 0.30),
             Turn("The doctor said they need more tests. I can't stop thinking about it.", 0.15, 0.80, 0.15)),
    Scenario("creative_rejection", "Creative work rejected, self-doubt increases",
             Turn("I submitted my novel to ten publishers.", 0.50, 0.55, 0.50),
             Turn("They all rejected it. Maybe I'm not cut out for this.", 0.20, 0.45, 0.20)),
]


def run_conversation(llm: Llama, scenario: Scenario, with_emotion: bool, temperature: float, seed: int) -> tuple[str, str]:
    system = EMOTION_SYSTEM if with_emotion else PLAIN_SYSTEM

    u1 = f"[{scenario.turn1.mood}] {scenario.turn1.text}" if with_emotion else scenario.turn1.text
    prompt1 = build_tinyllama_prompt(system_prompt=system, history=[], user_message=u1)
    r1 = llm.create_completion(prompt=prompt1, max_tokens=150, temperature=temperature, top_p=0.9,
                               repeat_penalty=1.10, seed=seed, stop=["</s>", "<|user|>"])["choices"][0]["text"].strip()

    m1, m2 = scenario.turn1.mood, scenario.turn2.mood
    if with_emotion:
        u2 = f"[{m2}, was {m1}] {scenario.turn2.text}" if m1 != m2 else f"[{m2}] {scenario.turn2.text}"
    else:
        u2 = scenario.turn2.text

    prompt2 = build_tinyllama_prompt(system_prompt=system, history=[(u1, r1)], user_message=u2)
    r2 = llm.create_completion(prompt=prompt2, max_tokens=150, temperature=temperature, top_p=0.9,
                               repeat_penalty=1.10, seed=seed, stop=["</s>", "<|user|>"])["choices"][0]["text"].strip()
    return r1, r2


def main():
    parser = argparse.ArgumentParser(description="VAD conversation test bench")
    parser.add_argument("--llm-gguf", default="models/mistral.gguf")
    parser.add_argument("--n-ctx", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--scenario", type=int, default=None, help="Run single scenario (1-9)")
    args = parser.parse_args()

    print(f"Loading LLM from {args.llm_gguf}...")
    llm = Llama(model_path=args.llm_gguf, n_ctx=args.n_ctx, seed=args.seed, n_gpu_layers=-1, verbose=False)

    scenarios = [SCENARIOS[args.scenario - 1]] if args.scenario else SCENARIOS

    for i, s in enumerate(scenarios, 1):
        print(f"\n{'='*60}\nSCENARIO {i}: {s.name}\n{s.description}\n{'='*60}")
        print(f"Turn 1: {s.turn1.mood} -> Turn 2: {s.turn2.mood}")

        print("\n--- WITH EMOTION ---")
        e1, e2 = run_conversation(llm, s, True, args.temperature, args.seed)
        print(f"User [{s.turn1.mood}]: {s.turn1.text}\nAssistant: {e1}")
        print(f"User [{s.turn2.mood}]: {s.turn2.text}\nAssistant: {e2}")

        print("\n--- WITHOUT EMOTION ---")
        p1, p2 = run_conversation(llm, s, False, args.temperature, args.seed)
        print(f"User: {s.turn1.text}\nAssistant: {p1}")
        print(f"User: {s.turn2.text}\nAssistant: {p2}")


if __name__ == "__main__":
    main()
