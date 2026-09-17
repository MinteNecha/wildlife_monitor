"""
Abstract Behaviour Model

Both LSTM and Transformer model implement the same interface
"""

from __future__ import annotations
from abc import ABC, abstractmethod
import torch
import torch.nn as nn

class BehaviourModel(ABC, nn.Module):
    @abstractmethod
    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run the model on a batch of padded sequences
        """

class LSTMBehaviourModel(BehaviourModel):
    """
    LSTM-based behaviour classifier
    """
    def __init__(self, num_features: int=5, hidden_size: int=128, num_layers: int=2, dropout: float=0.1, 
                 num_activity_classes: int=3, num_movement_classes: int=3,) -> None:
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.activity_head = nn.Linear(hidden_size, num_activity_classes)
        self.movement_head = nn.Linear(hidden_size, num_movement_classes)

    def forward(self, sequences: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            sequences, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        _, (final_hidden, _) = self.lstm(packed)

        last_layer_hidden = final_hidden[-1]

        activity_logits = self.activity_head(last_layer_hidden)
        movement_logits = self.movement_head(last_layer_hidden)

        return activity_logits, movement_logits