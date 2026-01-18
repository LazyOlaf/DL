"""
System prompts for emotion-aware conversation.
"""

EMOTION_AWARE_SYSTEM_PROMPT = """You are a conversational assistant.

You receive the user's emotional state as VAD values (0 to 1):
- Valence: 0 = negative, 1 = positive
- Arousal: 0 = calm, 1 = activated
- Dominance: 0 = submissive/overwhelmed, 1 = in-control

Be mindful of changes in VAD between turns. Respond naturally - do not mention or analyze emotions explicitly."""


NO_EMOTION_SYSTEM_PROMPT = """You are a conversational assistant. Respond naturally to what the user says."""


def format_emotion_context(
    valence: float,
    arousal: float,
    dominance: float,
    prev_valence: float | None = None,
    prev_arousal: float | None = None,
    prev_dominance: float | None = None,
) -> str:
    """
    Format VAD as a simple context string.
    Just the numbers - let the LLM interpret.
    """
    vad = f"V={valence:.2f}, A={arousal:.2f}, D={dominance:.2f}"

    # If we have previous values, show the change
    if prev_valence is not None:
        prev = f"prev: V={prev_valence:.2f}, A={prev_arousal:.2f}, D={prev_dominance:.2f}"
        return f"[VAD: {vad} | {prev}]"

    return f"[VAD: {vad}]"
