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


def classify_movement(
    camera_id: str,
    fidelity: pd.Series,
    territorial_percentile: float = 0.75,
) -> str:
    """
    Movement strategy based on cross-camera site fidelity: is this
    camera in the top 25% most favoured cameras for this species,
    compared to all other cameras that species was seen at.

    Uses a PERCENTILE threshold (not a fixed number) so it adjusts
    correctly regardless of how many cameras a species has - a fixed
    threshold was tested first and found not to generalise across
    species with different camera counts.

    A combined version also requiring time-clustering was tested and
    found to perform WORSE across all three species (buffalo, lion
    female, gazelle) in held-out evaluation - fidelity alone is kept
    as the simpler, better-performing version.

    territorial: fidelity in the top 25% of cameras for this species
    nomadic:     everything else

    Migratory detection is NOT attempted - this single-season dataset
    spans only ~108 real days, not a full year, so true seasonal
    migration cannot be reliably distinguished from the observation
    period simply ending.
    """
    threshold_value = fidelity.quantile(territorial_percentile)
    fid = fidelity.get(camera_id, 0.0)
    return "territorial" if fid >= threshold_value else "nomadic"

def build_training_set(
    detections_csv: str,
    max_length: int = 40,
) -> list[dict]:

    df = pd.read_csv(detections_csv)
    sequences = group_by_camera(df)
    fidelity = compute_site_fidelity(df)

    examples = []
    for camera_id, camera_df in sequences.items():
        padded_vectors, real_length, social_label = build_sequence(
            camera_df, max_length
        )
        activity_label = classify_activity(camera_df)
        movement_label = classify_movement(camera_id, fidelity)

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

def build_training_set_with_movement_variant(
    detections_csv: str,
    movement_variant: str,
    max_length: int = 40,
) -> list[dict]:
    """
    Same as build_training_set, but lets us pick which movement
    classification rule to test: "combined", "fidelity_only", or "70pct".
    """
    from wildlife_monitor.pipeline2.feature_extractor import extract_day_of_year

    df = pd.read_csv(detections_csv)
    sequences = group_by_camera(df)
    fidelity = compute_site_fidelity(df)

    all_days = [extract_day_of_year(ts) for ts in df["timestamp"]]
    dataset_span_days = max(all_days) - min(all_days)

    examples = []
    for camera_id, camera_df in sequences.items():
        padded_vectors, real_length, social_label = build_sequence(
            camera_df, max_length
        )
        activity_label = classify_activity(camera_df)

        if movement_variant == "combined":
            movement_label = classify_movement(
                camera_id, camera_df, fidelity, dataset_span_days
            )
        elif movement_variant == "fidelity_only":
            movement_label = classify_movement_fidelity_only(camera_id, fidelity)
        elif movement_variant == "70pct":
            movement_label = classify_movement_70pct(
                camera_id, camera_df, fidelity, dataset_span_days
            )
        else:
            raise ValueError(f"Unknown movement_variant: {movement_variant}")

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

def compute_site_fidelity(all_detections: pd.DataFrame) -> pd.Series:
    """
    For each camera, what fraction of this species' TOTAL detections
    (across every camera) happened here. A camera with a high fraction
    means this species is strongly concentrated there relative to
    everywhere else - a signal of site fidelity (territorial behaviour).
    A camera with a low, even fraction, similar to every other camera,
    suggests the species is spread out with no strong preference
    (nomadic behaviour).
    """
    total_detections = len(all_detections)
    camera_counts = all_detections.groupby("camera_id").size()
    return camera_counts / total_detections

def compute_seasonal_concentration(
    camera_sequence: pd.DataFrame, dataset_span_days: float
) -> float:
    """
    How tightly clustered a camera's detections are, relative to the
    REAL observed span of the whole dataset (not a full year - this
    dataset, Season 1, only covers ~108 real days, so true migratory
    detection is not achievable here; this measure is used only to
    strengthen the territorial/nomadic distinction).
    """
    from wildlife_monitor.pipeline2.feature_extractor import extract_day_of_year

    days = [extract_day_of_year(ts) for ts in camera_sequence["timestamp"]]

    if len(days) < 2:
        return 1.0

    span_days = max(days) - min(days)

    return 1.0 - (span_days / dataset_span_days)