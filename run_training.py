import argparse
from wildlife_monitor.pipeline2.models import LSTMBehaviourModel
from wildlife_monitor.pipeline2.train import (
    build_training_set, split_train_test, examples_to_tensors, train_model, evaluate_model
)

parser = argparse.ArgumentParser()
parser.add_argument("--species", required=True, help="e.g. buffalo, lionfemale, gazellethomsons")
parser.add_argument("--max_length", type=int, default=40)
args = parser.parse_args()

csv_path = f"results/bioclip_megadetector/detections_{args.species}.csv"

examples = build_training_set(csv_path, max_length=args.max_length)
train_examples, test_examples = split_train_test(examples)

train_sequences, train_lengths, train_activity, train_movement, train_month_features = examples_to_tensors(train_examples)
test_sequences, test_lengths, test_activity, test_movement, test_month_features = examples_to_tensors(test_examples)

print(f"Training LSTMBehaviourModel on {len(train_examples)} real {args.species} camera sequences...")
model = LSTMBehaviourModel()
trained_model = train_model(
    model, train_sequences, train_lengths, train_activity, train_movement, train_month_features
)

print("\n" + "="*60)
print(f"Evaluating on {len(test_examples)} held-out cameras (never seen during training):")
print("="*60)
evaluate_model(trained_model, test_sequences, test_lengths, test_activity, test_movement, test_month_features)