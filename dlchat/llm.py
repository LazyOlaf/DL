"""LLM wrapper and prompt formatting."""
from __future__ import annotations

from dataclasses import dataclass

from llama_cpp import Llama


@dataclass(frozen=True)
class LLMConfig:
    seed: int
    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    max_new_tokens: int


@dataclass(frozen=True)
class LLM:
    """Wrapper for llama-cpp-python."""
    model_path: str
    n_ctx: int
    seed: int

    def __post_init__(self) -> None:
        llm = Llama(
            model_path=self.model_path,
            n_ctx=self.n_ctx,
            seed=self.seed,
            n_gpu_layers=-1,
            verbose=False,
        )
        object.__setattr__(self, "_llm", llm)

    def count_tokens(self, text: str) -> int:
        tokens = self._llm.tokenize(text.encode("utf-8"), add_bos=False)
        return len(tokens)

    def complete(self, prompt: str, *, config: LLMConfig) -> str:
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=config.max_new_tokens,
            temperature=config.temperature,
            top_p=config.top_p,
            top_k=config.top_k,
            repeat_penalty=config.repeat_penalty,
            seed=config.seed,
            stop=["</s>", "<s>", "[INST]"],
        )
        return out["choices"][0]["text"].strip()


def build_prompt(
    *,
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
) -> str:
    """Build Mistral-style instruct prompt."""
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


def build_tinyllama_prompt(
    *,
    system_prompt: str,
    history: list[tuple[str, str]],
    user_message: str,
) -> str:
    """Build TinyLlama chat format prompt."""
    parts: list[str] = [f"<|system|>\n{system_prompt.strip()}</s>"]

    for u, a in history:
        parts.append(f"<|user|>\n{u.strip()}</s>")
        parts.append(f"<|assistant|>\n{a.strip()}</s>")

    parts.append(f"<|user|>\n{user_message.strip()}</s>")
    parts.append("<|assistant|>")

    return "\n".join(parts)
