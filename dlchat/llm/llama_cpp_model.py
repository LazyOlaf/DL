from __future__ import annotations

from dataclasses import dataclass

from llama_cpp import Llama

from dlchat.logging.schema import LLMDecoding


@dataclass(frozen=True)
class LlamaCppModel:
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

    def complete(self, prompt: str, *, decoding: LLMDecoding) -> str:
        out = self._llm.create_completion(
            prompt=prompt,
            max_tokens=decoding.max_new_tokens,
            temperature=decoding.temperature,
            top_p=decoding.top_p,
            top_k=decoding.top_k,
            repeat_penalty=decoding.repeat_penalty,
            seed=decoding.seed,
            stop=["</s>", "<s>", "[INST]"],
        )
        return out["choices"][0]["text"].strip()
