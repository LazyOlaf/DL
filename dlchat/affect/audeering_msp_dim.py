from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import Wav2Vec2Model, Wav2Vec2PreTrainedModel

from dlchat.logging.schema import AffectVAD


class _RegressionHead(nn.Module):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class _AudeeringEmotionModel(Wav2Vec2PreTrainedModel):
    def __init__(self, config) -> None:  # noqa: ANN001
        super().__init__(config)
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = _RegressionHead(config)
        self.init_weights()

    def forward(self, input_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        pooled = torch.mean(hidden_states, dim=1)
        logits = self.classifier(pooled)
        return pooled, logits


@dataclass(frozen=True)
class AudeeringMspDimRegressor:
    model_id: str

    def __post_init__(self) -> None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        object.__setattr__(self, "_device", device)
        object.__setattr__(self, "_processor", Wav2Vec2Processor.from_pretrained(self.model_id))
        model = _AudeeringEmotionModel.from_pretrained(self.model_id).to(device)
        model.eval()
        object.__setattr__(self, "_model", model)

    @torch.inference_mode()
    def predict(self, audio_f32: np.ndarray, *, sample_rate: int) -> AffectVAD:
        processed = self._processor(audio_f32, sampling_rate=sample_rate)
        x = processed["input_values"][0].reshape(1, -1)
        x = torch.from_numpy(x).to(self._device)

        _, logits = self._model(x)
        y = logits[0].detach().cpu().numpy()

        # Model card: outputs in approx 0..1, order is arousal/dominance/valence.
        arousal = float(np.clip(y[0], 0.0, 1.0))
        dominance = float(np.clip(y[1], 0.0, 1.0))
        valence = float(np.clip(y[2], 0.0, 1.0))

        return AffectVAD(valence=valence, arousal=arousal, dominance=dominance)
