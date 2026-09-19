"""
Training loop for Pipeline 2 behaviour models
"""
from __future__ import annotations
import torch 
import torch.nn as nn
import pandas as pd
import random

from wildlife_monitor.pipeline2.sequence_builder import group_by_camera, build_sequence
from wildlife_monitor.pipeline2.feature_extractor import extract_hour

ACTIVITY_CLASSES = ["diurnal", "nocturnal", "crepuscular"]
MOVEMENT_CLASSES = ["migratory", "territorial", "nomadic"]

def classify_activity(camera_sequence: pd.DataFrame) -> str:
    hours = [extract_hour(ts) for ts in camera_sequence["timestamp"]]

    categories = []
    for h in hours:
        if 6 <= h < 18:
            categories.append("diurnal")
        elif h < 5 or h >= 19:
            categories.append("nocturnal")
        else:
            categories.append("crepuscular")

    return max(set(categories), key=categories.count)

ACTIVITY_TO_IDX = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
MOVEMENT_TO_IDX = {name: i for i, name in enumerate(MOVEMENT_CLASSES)}


def classify_movement(camera_sequence: pd.DataFrame, threshold_days: float = 30.0) -> str:
    """
    Provisional movement strategy heuristic, based only on how spread
    out in time one camera's detections are. See project notes: this
    is the weakest of the three labels and a known limitation, since
    genuine movement strategy requires comparing patterns ACROSS
    camera sites, not just within one. "migratory" is intentionally
    not produced by this rule; only "territorial" or "nomadic".
    """
    timestamps = pd.to_datetime(camera_sequence["timestamp"], errors="coerce")
    timestamps = timestamps.dropna()

    if len(timestamps) < 2:
        return "territorial"

    span_days = (timestamps.max() - timestamps.min()).total_seconds() / 86400.0

    return "territorial" if span_days <= threshold_days else "nomadic"


def build_training_set(
    detections_csv: str,
    max_length: int = 40,
) -> list[dict]:
    """
    Read a Pipeline 1 detections CSV and build one training example
    per camera site, each with its padded sequence, real length, and
    the three labels (social structure, activity, movement).
    """
    df = pd.read_csv(detections_csv)
    sequences = group_by_camera(df)

    examples = []
    for camera_id, camera_df in sequences.items():
        padded_vectors, real_length, social_label = build_sequence(
            camera_df, max_length
        )
        activity_label = classify_activity(camera_df)
        movement_label = classify_movement(camera_df)

        examples.append({
            "camera_id": camera_id,
            "vectors": padded_vectors,
            "real_length": real_length,
            "social_label": social_label,
            "activity_label": activity_label,
            "activity_idx": ACTIVITY_TO_IDX[activity_label],
            "movement_label": movement_label,
            "movement_idx": MOVEMENT_TO_IDX[movement_label],
        })

    return examples

def split_train_test(examples: list[dict], test_fraction: float=0.2, seed: int=42,) -> tuple[list[dict], list[dict]]:
    """
    Randomly split camera examples into a training group and a testing group.
    """
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)

    num_test = max(1, round(len(shuffled)*test_fraction))
    test_examples = shuffled[:num_test]
    train_examples = shuffled[num_test:]

    return train_examples, test_examples

def examples_to_tensors(examples: list[dict],) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert a list of training examples into batched tensors.
    """
    sequences =  torch.tensor([ex["vectors"] for ex in examples], dtype=torch.float32)
    lengths = torch.tensor([ex["real_length"] for ex in examples], dtype=torch.int64)
    activity_idxs = torch.tensor([ex["activity_idx"] for ex in examples], dtype=torch.int64)
    movement_idxs = torch.tensor([ex["movement_idx"] for ex in examples], dtype=torch.int64)

    return sequences, lengths, activity_idxs, movement_idxs

def train_model(model: nn.Module, train_sequences: torch.Tensor, train_lengths: torch.Tensor, train_activity: torch.Tensor, train_movement: torch.Tensor,
                num_epochs: int=100, learning_rate: float=0.001,) -> nn.Module:
    """
    Training behaviour model on training given tensors
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()

    model.train()

    for epoch in range(num_epochs):
        optimizer.zero_grad()

        activity_logits, movement_logits = model(train_sequences, train_lengths)

        activity_loss= loss_function(activity_logits, train_activity)
        movement_loss = loss_function(movement_logits, train_movement)
        total_loss = activity_loss + movement_loss

        total_loss.backward()
        optimizer.step()

        if(epoch + 1)%10 == 0:
            print(f"Epoch {epoch+1:>3}/{num_epochs} "
                  f"loss={total_loss.item():.4f}  "
                  f"(activity={activity_loss.item():.4f}, "
                  f"movement={movement_loss.item():.4f})")

    return model

def evaluate_model(
    model: nn.Module,
    test_sequences: torch.Tensor,
    test_lengths: torch.Tensor,
    test_activity: torch.Tensor,
    test_movement: torch.Tensor,
) -> None:
    """
    Check the trained model's performance on held-out data it never
    saw during training.
    """
    model.eval()

    with torch.no_grad():
        activity_logits, movement_logits = model(test_sequences, test_lengths)

    activity_preds = activity_logits.argmax(dim=1)
    movement_preds = movement_logits.argmax(dim=1)

    print("Activity predictions vs true labels:")
    for i in range(len(test_activity)):
        true_label = ACTIVITY_CLASSES[test_activity[i].item()]
        pred_label = ACTIVITY_CLASSES[activity_preds[i].item()]
        correct = "correct" if true_label == pred_label else "WRONG"
        print(f"  true={true_label:<12} predicted={pred_label:<12} [{correct}]")

    activity_accuracy = (activity_preds == test_activity).float().mean().item()
    print(f"\nOverall activity accuracy: {activity_accuracy*100:.1f}%")

    print("\nMovement predictions vs true labels:")
    for i in range(len(test_movement)):
        true_label = MOVEMENT_CLASSES[test_movement[i].item()]
        pred_label = MOVEMENT_CLASSES[movement_preds[i].item()]
        correct = "correct" if true_label == pred_label else "WRONG"
        print(f"  true={true_label:<12} predicted={pred_label:<12} [{correct}]")

    movement_accuracy = (movement_preds == test_movement).float().mean().item()
    print(f"\nOverall movement accuracy: {movement_accuracy*100:.1f}%")