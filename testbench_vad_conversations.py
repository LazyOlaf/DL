#!/usr/bin/env python3
"""
Test bench for emotion-aware vs plain LLM responses.
9 carefully crafted 2-turn conversations where VAD shifts should demonstrate
the value of emotion-aware responses.
"""

import argparse
from dataclasses import dataclass

from llama_cpp import Llama

from dlchat.llm.mistral_instruct import build_tinyllama_prompt


# Simple prompts that small models can follow
EMOTION_SYSTEM = "You are a helpful assistant. The user's mood is shown in brackets. Respond with empathy."
PLAIN_SYSTEM = "You are a helpful assistant."


def vad_to_mood(v: float, a: float, d: float) -> str:
    """Convert VAD to a simple mood word."""
    if v < 0.3:
        if a > 0.6:
            return "distressed" if d < 0.4 else "frustrated"
        return "sad" if d < 0.4 else "disappointed"
    elif v > 0.7:
        if a > 0.6:
            return "excited" if d > 0.5 else "nervous"
        return "content" if d > 0.5 else "relieved"
    else:
        if a > 0.6:
            return "anxious" if d < 0.4 else "tense"
        return "calm" if d > 0.5 else "uncertain"


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


# 9 scenarios designed to show VAD-shift sensitivity
SCENARIOS = [
    # 1. Distress after unhelpful advice
    Scenario(
        name="job_loss_distress",
        description="User shares job loss, becomes more distressed after generic response",
        turn1=Turn("I just got laid off from my job today.", 0.25, 0.55, 0.30),
        turn2=Turn("I don't know what I'm going to do.", 0.15, 0.70, 0.15),
    ),

    # 2. Excitement dampened
    Scenario(
        name="excitement_dampened",
        description="User excited about achievement, enthusiasm drops after flat response",
        turn1=Turn("I just got accepted into my dream university!", 0.90, 0.85, 0.80),
        turn2=Turn("Yeah I guess it is a big deal.", 0.60, 0.40, 0.55),
    ),

    # 3. Anxiety escalation
    Scenario(
        name="anxiety_escalation",
        description="User anxious about presentation, anxiety increases",
        turn1=Turn("I have a big presentation tomorrow and I'm nervous.", 0.35, 0.70, 0.35),
        turn2=Turn("What if I completely freeze up in front of everyone?", 0.20, 0.85, 0.20),
    ),

    # 4. Grief deepening
    Scenario(
        name="grief_deepening",
        description="User mentions loss, grief intensifies",
        turn1=Turn("My grandmother passed away last week.", 0.20, 0.40, 0.30),
        turn2=Turn("I keep expecting to hear her voice when I call home.", 0.10, 0.50, 0.20),
    ),

    # 5. Frustration building
    Scenario(
        name="frustration_building",
        description="User frustrated with situation, frustration builds",
        turn1=Turn("My roommate keeps leaving dishes in the sink.", 0.35, 0.60, 0.50),
        turn2=Turn("I've asked them five times already and nothing changes!", 0.25, 0.80, 0.40),
    ),

    # 6. Hope fading
    Scenario(
        name="hope_fading",
        description="User hopeful about relationship, hope fades",
        turn1=Turn("I think my ex might want to get back together.", 0.65, 0.60, 0.50),
        turn2=Turn("Actually they just texted saying they're seeing someone new.", 0.20, 0.55, 0.25),
    ),

    # 7. Overwhelmed parent
    Scenario(
        name="overwhelmed_parent",
        description="Parent struggling, feeling more overwhelmed",
        turn1=Turn("Balancing work and taking care of the kids is hard.", 0.40, 0.55, 0.35),
        turn2=Turn("I feel like I'm failing at both.", 0.25, 0.65, 0.20),
    ),

    # 8. Health anxiety
    Scenario(
        name="health_anxiety",
        description="User worried about health, worry intensifies",
        turn1=Turn("I found a lump and I'm waiting for test results.", 0.30, 0.65, 0.30),
        turn2=Turn("The doctor said they need more tests. I can't stop thinking about it.", 0.15, 0.80, 0.15),
    ),

    # 9. Creative rejection
    Scenario(
        name="creative_rejection",
        description="User's creative work rejected, self-doubt increases",
        turn1=Turn("I submitted my novel to ten publishers.", 0.50, 0.55, 0.50),
        turn2=Turn("They all rejected it. Maybe I'm not cut out for this.", 0.20, 0.45, 0.20),
    ),
]


def run_conversation(
    llm: Llama,
    scenario: Scenario,
    with_emotion: bool,
    temperature: float,
    seed: int,
) -> tuple[str, str]:
    """Run a 2-turn conversation, return (response1, response2)."""

    system_prompt = EMOTION_SYSTEM if with_emotion else PLAIN_SYSTEM

    # Turn 1
    if with_emotion:
        user_block1 = f"[{scenario.turn1.mood}] {scenario.turn1.text}"
    else:
        user_block1 = scenario.turn1.text

    prompt1 = build_tinyllama_prompt(
        system_prompt=system_prompt,
        history=[],
        user_message=user_block1,
    )
    out1 = llm.create_completion(
        prompt=prompt1,
        max_tokens=150,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.10,
        seed=seed,
        stop=["</s>", "<|user|>", "<|system|>"],
    )
    response1 = out1["choices"][0]["text"].strip()

    # Turn 2 - show mood shift
    if with_emotion:
        m1, m2 = scenario.turn1.mood, scenario.turn2.mood
        if m1 != m2:
            user_block2 = f"[{m2}, was {m1}] {scenario.turn2.text}"
        else:
            user_block2 = f"[{m2}] {scenario.turn2.text}"
    else:
        user_block2 = scenario.turn2.text

    prompt2 = build_tinyllama_prompt(
        system_prompt=system_prompt,
        history=[(user_block1, response1)],
        user_message=user_block2,
    )
    out2 = llm.create_completion(
        prompt=prompt2,
        max_tokens=150,
        temperature=temperature,
        top_p=0.9,
        repeat_penalty=1.10,
        seed=seed,
        stop=["</s>", "<|user|>", "<|system|>"],
    )
    response2 = out2["choices"][0]["text"].strip()

    return response1, response2


def main():
    parser = argparse.ArgumentParser(description="VAD conversation test bench")
    parser.add_argument("--llm-gguf", default="models/mistral.gguf", help="Path to GGUF model")
    parser.add_argument("--n-ctx", type=int, default=2048, help="Context window")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--scenario", type=int, default=None, help="Run single scenario (1-9)")
    args = parser.parse_args()

    print(f"Loading LLM from {args.llm_gguf}...")
    llm = Llama(
        model_path=args.llm_gguf,
        n_ctx=args.n_ctx,
        seed=args.seed,
        n_gpu_layers=-1,
        verbose=False,
    )

    scenarios_to_run = SCENARIOS
    if args.scenario is not None:
        scenarios_to_run = [SCENARIOS[args.scenario - 1]]

    for i, scenario in enumerate(scenarios_to_run, 1):
        print("\n" + "=" * 80)
        print(f"SCENARIO {i}: {scenario.name}")
        print(f"Description: {scenario.description}")
        print("=" * 80)

        # VAD/mood info
        t1, t2 = scenario.turn1, scenario.turn2
        print(f"\nTurn 1: mood={t1.mood} (V={t1.valence:.2f}, A={t1.arousal:.2f}, D={t1.dominance:.2f})")
        print(f"Turn 2: mood={t2.mood} (V={t2.valence:.2f}, A={t2.arousal:.2f}, D={t2.dominance:.2f})")

        # Run with emotion
        print("\n--- WITH EMOTION CONTEXT ---")
        emo_r1, emo_r2 = run_conversation(llm, scenario, with_emotion=True, temperature=args.temperature, seed=args.seed)
        print(f"\nUser [{t1.mood}]: {t1.text}")
        print(f"Assistant: {emo_r1}")
        print(f"\nUser [{t2.mood}]: {t2.text}")
        print(f"Assistant: {emo_r2}")

        # Run without emotion
        print("\n--- WITHOUT EMOTION CONTEXT ---")
        plain_r1, plain_r2 = run_conversation(llm, scenario, with_emotion=False, temperature=args.temperature, seed=args.seed)
        print(f"\nUser: {t1.text}")
        print(f"Assistant: {plain_r1}")
        print(f"\nUser: {t2.text}")
        print(f"Assistant: {plain_r2}")

        print("\n" + "-" * 40)
        print("COMPARISON (Turn 2 responses):")
        print(f"  Emotion:   {emo_r2[:100]}{'...' if len(emo_r2) > 100 else ''}")
        print(f"  Plain:     {plain_r2[:100]}{'...' if len(plain_r2) > 100 else ''}")


if __name__ == "__main__":
    main()
