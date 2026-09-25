import random
import gc
import numpy as np
import pandas as pd

from wildlife_monitor.pipeline2.autodiff import Adam, softmax_cross_entropy, clip_grad_norm
from wildlife_monitor.pipeline2.autodiff import Adam, softmax_cross_entropy
from wildlife_monitor.pipeline2.sequence_builder import (
    group_by_camera, build_sequence, compute_site_fidelity,
)

ACTIVITY_CLASSES = ["diurnal", "nocturnal", "crepuscular"]
MOVEMENT_CLASSES = ["migratory", "territorial", "nomadic"]
ACTIVITY_TO_IDX = {name: i for i, name in enumerate(ACTIVITY_CLASSES)}
MOVEMENT_TO_IDX = {name: i for i, name in enumerate(MOVEMENT_CLASSES)}


def extract_hour(timestamp):
    from wildlife_monitor.pipeline2.feature_extractor import extract_hour as _extract_hour
    return _extract_hour(timestamp)


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


def classify_movement(camera_id: str, fidelity: pd.Series,
                       territorial_percentile: float = 0.75) -> str:
    threshold_value = fidelity.quantile(territorial_percentile)
    fid = fidelity.get(camera_id, 0.0)
    return "territorial" if fid >= threshold_value else "nomadic"


def build_training_set(detections_csv: str, max_length: int = 40) -> list:
    df = pd.read_csv(detections_csv)
    sequences = group_by_camera(df)
    fidelity = compute_site_fidelity(df)
    examples = []
    for camera_id, camera_df in sequences.items():
        padded_vectors, real_length, social_label = build_sequence(camera_df, max_length)
        activity_label = classify_activity(camera_df)
        movement_label = classify_movement(camera_id, fidelity)
        examples.append({
            "camera_id": camera_id,
            "vectors": padded_vectors,
            "real_length": real_length,
            "social_label": social_label,
            "activity_label": activity_label,
            "movement_label": movement_label,
            "activity_idx": ACTIVITY_TO_IDX[activity_label],
            "movement_idx": MOVEMENT_TO_IDX[movement_label],
        })
    return examples


def split_train_test(examples, test_fraction=0.2, seed=42):
    shuffled = examples.copy()
    random.Random(seed).shuffle(shuffled)
    num_test = max(1, round(len(shuffled) * test_fraction))
    return shuffled[num_test:], shuffled[:num_test]


def examples_to_tensors(examples):
    sequences = np.array([ex["vectors"] for ex in examples], dtype=np.float64)
    lengths = np.array([ex["real_length"] for ex in examples], dtype=np.int64)
    activity_idxs = np.array([ex["activity_idx"] for ex in examples], dtype=np.int64)
    movement_idxs = np.array([ex["movement_idx"] for ex in examples], dtype=np.int64)
    return sequences, lengths, activity_idxs, movement_idxs


def train_model(model, train_sequences, train_lengths, train_activity, train_movement,
                 num_epochs=100, learning_rate=0.001, max_grad_norm=5.0):
    optimizer = Adam(model.parameters(), lr=learning_rate)
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        activity_logits, movement_logits = model.forward(train_sequences, train_lengths, training=True)
        activity_loss = softmax_cross_entropy(activity_logits, train_activity)
        movement_loss = softmax_cross_entropy(movement_logits, train_movement)
        total_loss = activity_loss + movement_loss
        total_loss.backward()
        clip_grad_norm(model.parameters(), max_grad_norm)
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch + 1:>3}/{num_epochs}  loss={total_loss.data:.4f}  "
                  f"(activity={activity_loss.data:.4f}, movement={movement_loss.data:.4f})")
            gc.collect()
    return model


def evaluate_model(model, test_sequences, test_lengths, test_activity, test_movement):
    activity_logits, movement_logits = model.forward(test_sequences, test_lengths)
    activity_preds = np.argmax(activity_logits.data, axis=1)
    movement_preds = np.argmax(movement_logits.data, axis=1)

    print("Activity predictions vs true labels:")
    activity_correct = 0
    for true_idx, pred_idx in zip(test_activity, activity_preds):
        status = "[correct]" if true_idx == pred_idx else "[WRONG]"
        print(f"  true={ACTIVITY_CLASSES[true_idx]:<12} predicted={ACTIVITY_CLASSES[pred_idx]:<12} {status}")
        activity_correct += int(true_idx == pred_idx)
    activity_accuracy = activity_correct / len(test_activity) * 100
    print(f"\nOverall activity accuracy: {activity_accuracy:.1f}%\n")

    print("Movement predictions vs true labels:")
    movement_correct = 0
    for true_idx, pred_idx in zip(test_movement, movement_preds):
        status = "[correct]" if true_idx == pred_idx else "[WRONG]"
        print(f"  true={MOVEMENT_CLASSES[true_idx]:<12} predicted={MOVEMENT_CLASSES[pred_idx]:<12} {status}")
        movement_correct += int(true_idx == pred_idx)
    movement_accuracy = movement_correct / len(test_movement) * 100
    print(f"\nOverall movement accuracy: {movement_accuracy:.1f}%")
    return activity_accuracy, movement_accuracy