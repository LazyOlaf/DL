from __future__ import annotations

from dlchat.logging.schema import AffectVAD

NEUTRAL_BAND = 0.05
SLIGHT_BAND = 0.15
MODERATE_BAND = 0.30


def _intensity_word(value_01: float) -> str:
    d = abs(value_01 - 0.5)
    if d <= NEUTRAL_BAND:
        return "neutral"
    if d <= SLIGHT_BAND:
        return "slightly"
    if d <= MODERATE_BAND:
        return "moderately"
    return "strongly"


def _axis_phrase(*, axis: str, value_01: float) -> str:
    intensity = _intensity_word(value_01)
    delta = value_01 - 0.5

    if axis == "valence":
        if intensity == "neutral":
            return (
                f"valence: neutral ({value_01:.2f}; close to mid-point, neither clearly pleasant nor unpleasant)"
            )
        if delta > 0:
            return (
                f"valence: {intensity} positive ({value_01:.2f}; pleasant/rewarding, approach-leaning, positive affect)"
            )
        return (
            f"valence: {intensity} negative ({value_01:.2f}; unpleasant/aversive, withdrawal-leaning, negative affect)"
        )

    if axis == "arousal":
        if intensity == "neutral":
            return (
                f"arousal: neutral ({value_01:.2f}; close to mid-point, neither clearly calm nor clearly activated)"
            )
        if delta > 0:
            return (
                f"arousal: {intensity} high ({value_01:.2f}; activated/energetic, higher physiological activation, potentially tense)"
            )
        return (
            f"arousal: {intensity} low ({value_01:.2f}; calm/relaxed, lower physiological activation, possibly tired)"
        )

    if axis == "dominance":
        if intensity == "neutral":
            return (
                f"dominance: neutral ({value_01:.2f}; close to mid-point, neither clearly in-control nor clearly uncertain)"
            )
        if delta > 0:
            return (
                f"dominance: {intensity} high ({value_01:.2f}; confident/assertive, higher perceived control/agency)"
            )
        return (
            f"dominance: {intensity} low ({value_01:.2f}; uncertain/submissive-leaning, lower perceived control/agency)"
        )

    raise ValueError(f"unknown axis: {axis}")


def _composite_phrase(vad: AffectVAD) -> str | None:
    v_int = _intensity_word(vad.valence)
    a_int = _intensity_word(vad.arousal)
    if v_int == "neutral" or a_int == "neutral":
        return None

    v_pos = vad.valence > 0.5
    a_high = vad.arousal > 0.5

    if (not v_pos) and a_high:
        return "composite: stress-leaning (negative/aversive valence + high activation/arousal)"
    if (not v_pos) and (not a_high):
        return "composite: down/withdrawn-leaning (negative/aversive valence + low activation/arousal)"
    if v_pos and a_high:
        return "composite: excited/engaged-leaning (positive/pleasant valence + high activation/arousal)"
    if v_pos and (not a_high):
        return "composite: content/at-ease-leaning (positive/pleasant valence + low activation/arousal)"

    return None


def describe_vad(vad: AffectVAD) -> list[str]:
    desc = [
        _axis_phrase(axis="valence", value_01=vad.valence),
        _axis_phrase(axis="arousal", value_01=vad.arousal),
        _axis_phrase(axis="dominance", value_01=vad.dominance),
    ]
    comp = _composite_phrase(vad)
    if comp is not None:
        desc.append(comp)
    return desc
