from __future__ import annotations


def build_mistral_instruct_prompt(
    *,
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
) -> str:
    parts: list[str] = []

    for i, (u, a) in enumerate(history):
        if i == 0:
            u = system_prompt.rstrip() + "\n\n" + u.lstrip()
        parts.append(f"<s>[INST] {u.strip()} [/INST] {a.strip()}</s>")

    if history:
        u_last = user_message
    else:
        u_last = system_prompt.rstrip() + "\n\n" + user_message.lstrip()

    parts.append(f"<s>[INST] {u_last.strip()} [/INST]")
    return "".join(parts)
