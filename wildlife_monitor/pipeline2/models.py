"""
Behavioural models (Package P3).

Two architectures satisfy FR5: an LSTM and a Transformer encoder, both built
on the project's own autodiff engine rather than a deep-learning framework.
They share the :class:`BehaviourModel` interface, so the training script, the
inference service and the dashboard treat them interchangeably.

Checkpointing lives on the base class. A checkpoint is a single ``.npz``
holding every named parameter plus a ``__meta__`` JSON blob describing the
architecture, the label taxonomy and the training run. That is enough to
rebuild a model from disk without the caller knowing which architecture it is.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

from wildlife_monitor.pipeline2.autodiff import Tensor, layernorm
from wildlife_monitor.pipeline2.feature_extractor import NUM_FEATURES

CHECKPOINT_VERSION = 1


def init_weight(shape):
    fan_in = shape[0]
    return Tensor(np.random.randn(*shape) / np.sqrt(fan_in))


class BehaviourModel(ABC):
    """Interface shared by every behavioural architecture."""

    @abstractmethod
    def forward(self, sequences: np.ndarray, lengths: np.ndarray,
                month_features: np.ndarray, training: bool = True):
        """Returns (activity_logits, movement_logits) as Tensors."""

    @abstractmethod
    def named_parameters(self) -> list[tuple[str, Tensor]]:
        """Every trainable tensor, paired with a stable name."""

    @abstractmethod
    def config(self) -> dict[str, Any]:
        """Constructor arguments needed to rebuild this model from disk."""

    def parameters(self) -> list[Tensor]:
        """Trainable tensors, in the order the optimiser expects."""
        return [tensor for _, tensor in self.named_parameters()]

    # ── Persistence ───────────────────────────────────────────────────────────
    def save_checkpoint(self, path: str | Path,
                        metadata: dict[str, Any] | None = None) -> Path:
        """Write parameters plus architecture metadata to a single ``.npz``."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {name: tensor.data for name, tensor in self.named_parameters()}
        meta = {
            "checkpoint_version": CHECKPOINT_VERSION,
            "model_type": type(self).__name__,
            "config": self.config(),
            **(metadata or {}),
        }
        arrays["__meta__"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        return path

    def load_state(self, arrays: dict[str, np.ndarray]) -> None:
        """Copy saved arrays back into this model's parameters, by name."""
        for name, tensor in self.named_parameters():
            if name not in arrays:
                raise KeyError(f"Checkpoint is missing parameter '{name}'")
            saved = arrays[name]
            if saved.shape != tensor.data.shape:
                raise ValueError(
                    f"Shape mismatch for '{name}': checkpoint has {saved.shape}, "
                    f"model expects {tensor.data.shape}")
            tensor.data = saved.astype(np.float64)


class LSTMBehaviourModel(BehaviourModel):
    def __init__(self, num_features=NUM_FEATURES, hidden_size=128, num_layers=2, dropout=0.1,
                 num_activity_classes=3, num_movement_classes=3, num_month_features=12):
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_activity_classes = num_activity_classes
        self.num_movement_classes = num_movement_classes
        self.num_month_features = num_month_features

        self.layers = []
        in_dim = num_features
        for _ in range(num_layers):
            self.layers.append({
                "W_i": init_weight((hidden_size + in_dim, hidden_size)),
                "W_f": init_weight((hidden_size + in_dim, hidden_size)),
                "W_o": init_weight((hidden_size + in_dim, hidden_size)),
                "W_g": init_weight((hidden_size + in_dim, hidden_size)),
                "b_i": Tensor(np.zeros(hidden_size)),
                "b_f": Tensor(np.ones(hidden_size)),
                "b_o": Tensor(np.zeros(hidden_size)),
                "b_g": Tensor(np.zeros(hidden_size)),
            })
            in_dim = hidden_size

        self.W_activity = init_weight((hidden_size, num_activity_classes))
        self.b_activity = Tensor(np.zeros(num_activity_classes))
        # The movement head also reads the per-camera monthly detection-share
        # vector, computed from that camera's full (uncapped) season, so it
        # sees seasonal shape even when the raw sequence was truncated.
        self.W_movement = init_weight((hidden_size + num_month_features,
                                        num_movement_classes))
        self.b_movement = Tensor(np.zeros(num_movement_classes))

    def config(self) -> dict[str, Any]:
        return {
            "num_features": self.num_features,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "num_activity_classes": self.num_activity_classes,
            "num_movement_classes": self.num_movement_classes,
            "num_month_features": self.num_month_features,
        }

    def named_parameters(self) -> list[tuple[str, Tensor]]:
        named = [
            ("head.activity.W", self.W_activity),
            ("head.activity.b", self.b_activity),
            ("head.movement.W", self.W_movement),
            ("head.movement.b", self.b_movement),
        ]
        for index, layer in enumerate(self.layers):
            named.extend((f"lstm.{index}.{key}", tensor)
                          for key, tensor in layer.items())
        return named

    def forward(self, sequences, lengths, month_features, training=True):
        batch, seq_len, _ = sequences.shape
        mask = (np.arange(seq_len)[None, :] < lengths[:, None]).astype(np.float64)
        keep_mask = 1.0 - mask

        layer_input = [Tensor(sequences[:, t, :]) for t in range(seq_len)]
        for layer_idx, layer in enumerate(self.layers):
            h = Tensor(np.zeros((batch, self.hidden_size)))
            c = Tensor(np.zeros((batch, self.hidden_size)))
            outputs = []
            for t in range(seq_len):
                x_t = layer_input[t]
                z = Tensor.concat([h, x_t])
                i = (z @ layer["W_i"] + layer["b_i"]).sigmoid()
                f = (z @ layer["W_f"] + layer["b_f"]).sigmoid()
                o = (z @ layer["W_o"] + layer["b_o"]).sigmoid()
                g = (z @ layer["W_g"] + layer["b_g"]).tanh()
                c_new = f * c + i * g
                h_new = o * c_new.tanh()

                m_t = Tensor(mask[:, t:t + 1])
                k_t = Tensor(keep_mask[:, t:t + 1])
                c = m_t * c_new + k_t * c
                h = m_t * h_new + k_t * h
                outputs.append(h)

            if layer_idx < len(self.layers) - 1:
                outputs = [o.dropout(self.dropout, training=training) for o in outputs]
            layer_input = outputs

        final_hidden = layer_input[-1]
        activity_logits = final_hidden @ self.W_activity + self.b_activity
        movement_input = Tensor.concat([final_hidden, Tensor(month_features)], axis=-1)
        movement_logits = movement_input @ self.W_movement + self.b_movement
        return activity_logits, movement_logits


class TransformerBehaviourModel(BehaviourModel):
    def __init__(self, num_features=NUM_FEATURES, hidden_size=128, num_heads=4, num_layers=3,
                 max_length=30, ff_hidden=256, dropout=0.1, activation="relu",
                 num_activity_classes=3, num_movement_classes=3, num_month_features=12):
        assert hidden_size % num_heads == 0
        self.num_features = num_features
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads
        self.num_layers = num_layers
        self.max_length = max_length
        self.ff_hidden = ff_hidden
        self.dropout = dropout
        self.activation = activation
        self.num_activity_classes = num_activity_classes
        self.num_movement_classes = num_movement_classes
        self.num_month_features = num_month_features

        self.W_proj = init_weight((num_features, hidden_size))
        self.b_proj = Tensor(np.zeros(hidden_size))
        self.pos_enc = Tensor(np.random.randn(1, max_length, hidden_size) * 0.01)

        self.blocks = []
        for _ in range(num_layers):
            self.blocks.append({
                "Wq": [init_weight((hidden_size, self.d_head)) for _ in range(num_heads)],
                "Wk": [init_weight((hidden_size, self.d_head)) for _ in range(num_heads)],
                "Wv": [init_weight((hidden_size, self.d_head)) for _ in range(num_heads)],
                "Wo": init_weight((hidden_size, hidden_size)),
                "bo": Tensor(np.zeros(hidden_size)),
                "gain1": Tensor(np.ones(hidden_size)),
                "bias1": Tensor(np.zeros(hidden_size)),
                "W_ff1": init_weight((hidden_size, ff_hidden)),
                "b_ff1": Tensor(np.zeros(ff_hidden)),
                "W_ff2": init_weight((ff_hidden, hidden_size)),
                "b_ff2": Tensor(np.zeros(hidden_size)),
                "gain2": Tensor(np.ones(hidden_size)),
                "bias2": Tensor(np.zeros(hidden_size)),
            })

        self.W_activity = init_weight((hidden_size, num_activity_classes))
        self.b_activity = Tensor(np.zeros(num_activity_classes))
        self.W_movement = init_weight((hidden_size + num_month_features,
                                        num_movement_classes))
        self.b_movement = Tensor(np.zeros(num_movement_classes))

    def config(self) -> dict[str, Any]:
        return {
            "num_features": self.num_features,
            "hidden_size": self.hidden_size,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "max_length": self.max_length,
            "ff_hidden": self.ff_hidden,
            "dropout": self.dropout,
            "activation": self.activation,
            "num_activity_classes": self.num_activity_classes,
            "num_movement_classes": self.num_movement_classes,
            "num_month_features": self.num_month_features,
        }

    def named_parameters(self) -> list[tuple[str, Tensor]]:
        named = [
            ("proj.W", self.W_proj),
            ("proj.b", self.b_proj),
            ("pos_enc", self.pos_enc),
            ("head.activity.W", self.W_activity),
            ("head.activity.b", self.b_activity),
            ("head.movement.W", self.W_movement),
            ("head.movement.b", self.b_movement),
        ]
        for index, block in enumerate(self.blocks):
            for key, value in block.items():
                if isinstance(value, list):
                    named.extend((f"block.{index}.{key}.{head}", tensor)
                                  for head, tensor in enumerate(value))
                else:
                    named.append((f"block.{index}.{key}", value))
        return named

    def forward(self, sequences, lengths, month_features, training=True):
        batch, seq_len, _ = sequences.shape
        mask_1d = (np.arange(seq_len)[None, :] < lengths[:, None]).astype(np.float64)
        key_mask_bias = np.where(mask_1d[:, None, :] > 0, 0.0, -1e9)

        x = Tensor.stack([Tensor(sequences[:, t, :]) for t in range(seq_len)])
        x = x @ self.W_proj + self.b_proj
        x = x + self.pos_enc

        for block in self.blocks:
            heads_out = []
            for h in range(self.num_heads):
                Q = x @ block["Wq"][h]
                K = x @ block["Wk"][h]
                V = x @ block["Wv"][h]
                scores = (Q @ K.transpose_last2()) * (1.0 / np.sqrt(self.d_head))
                scores = scores + Tensor(key_mask_bias)
                attn = scores.softmax()
                heads_out.append(attn @ V)
            attn_out = Tensor.concat(heads_out) @ block["Wo"] + block["bo"]
            attn_out = attn_out.dropout(self.dropout, training=training)

            x1 = layernorm(x + attn_out, block["gain1"], block["bias1"])
            if self.activation == "gelu":
                ff = (x1 @ block["W_ff1"] + block["b_ff1"]).gelu()
            else:
                ff = (x1 @ block["W_ff1"] + block["b_ff1"]).relu()
            ff = ff @ block["W_ff2"] + block["b_ff2"]
            ff = ff.dropout(self.dropout, training=training)
            x = layernorm(x1 + ff, block["gain2"], block["bias2"])

        mask_3d = Tensor(mask_1d[:, :, None])
        summed = (x * mask_3d).sum_axis(axis=1)
        lengths_t = Tensor(lengths[:, None].astype(np.float64))
        pooled = summed / lengths_t

        activity_logits = pooled @ self.W_activity + self.b_activity
        movement_input = Tensor.concat([pooled, Tensor(month_features)], axis=-1)
        movement_logits = movement_input @ self.W_movement + self.b_movement
        return activity_logits, movement_logits


ARCHITECTURES: dict[str, type[BehaviourModel]] = {
    "lstm": LSTMBehaviourModel,
    "transformer": TransformerBehaviourModel,
    "LSTMBehaviourModel": LSTMBehaviourModel,
    "TransformerBehaviourModel": TransformerBehaviourModel,
}


def build_model(architecture: str, **overrides) -> BehaviourModel:
    """Construct a model by short name ('lstm' or 'transformer')."""
    key = architecture.strip().lower()
    if key not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{architecture}'. "
                          f"Expected one of: lstm, transformer")
    return ARCHITECTURES[key](**overrides)


def load_checkpoint(path: str | Path) -> tuple[BehaviourModel, dict[str, Any]]:
    """Rebuild a model from a checkpoint, returning it with its metadata."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No behaviour checkpoint at {path}")

    with np.load(path, allow_pickle=False) as data:
        meta = json.loads(str(data["__meta__"]))
        arrays = {key: data[key] for key in data.files if key != "__meta__"}

    model_type = meta.get("model_type", "")
    if model_type not in ARCHITECTURES:
        raise ValueError(f"Checkpoint names unknown architecture '{model_type}'")

    model = ARCHITECTURES[model_type](**meta.get("config", {}))
    model.load_state(arrays)
    return model, meta
