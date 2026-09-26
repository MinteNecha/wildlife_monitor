from abc import ABC, abstractmethod

import numpy as np

from wildlife_monitor.pipeline2.autodiff import Tensor, layernorm


def init_weight(shape):
    fan_in = shape[0]
    return Tensor(np.random.randn(*shape) / np.sqrt(fan_in))


class BehaviourModel(ABC):
    @abstractmethod
    def forward(self, sequences: np.ndarray, lengths: np.ndarray, month_features: np.ndarray, training: bool = True):
        """Returns (activity_logits, movement_logits) as Tensors."""

    @abstractmethod
    def parameters(self):
        """Returns a list of trainable Tensor parameters."""


class LSTMBehaviourModel(BehaviourModel):
    def __init__(self, num_features=5, hidden_size=128, num_layers=2, dropout=0.1,
                 num_activity_classes=3, num_movement_classes=3, num_month_features=12):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
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
        # Movement head also takes the per-camera monthly detection-share
        # vector, computed from that camera's full (uncapped) season, so it
        # sees the seasonal shape even for cameras whose raw sequence is
        # truncated to max_length.
        self.W_movement = init_weight((hidden_size + num_month_features, num_movement_classes))
        self.b_movement = Tensor(np.zeros(num_movement_classes))

    def parameters(self):
        params = [self.W_activity, self.b_activity, self.W_movement, self.b_movement]
        for layer in self.layers:
            params.extend(layer.values())
        return params

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
    def __init__(self, num_features=5, hidden_size=128, num_heads=4, num_layers=3,
                 max_length=30, ff_hidden=256, dropout=0.1, activation="relu",
                 num_activity_classes=3, num_movement_classes=3, num_month_features=12):
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.d_head = hidden_size // num_heads
        self.max_length = max_length
        self.dropout = dropout
        self.activation = activation

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
        # Movement head also takes the per-camera monthly detection-share
        # vector, computed from that camera's full (uncapped) season, so it
        # sees the seasonal shape even for cameras whose raw sequence is
        # truncated to max_length.
        self.W_movement = init_weight((hidden_size + num_month_features, num_movement_classes))
        self.b_movement = Tensor(np.zeros(num_movement_classes))

    def parameters(self):
        params = [self.W_proj, self.b_proj, self.pos_enc,
                  self.W_activity, self.b_activity, self.W_movement, self.b_movement]
        for block in self.blocks:
            for val in block.values():
                params.extend(val) if isinstance(val, list) else params.append(val)
        return params

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