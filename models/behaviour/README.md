# Behaviour model checkpoints

Trained Pipeline 2 models are written here as `<species>_<architecture>.npz`
by `scripts/train_behaviour.py`.

Each file holds every named parameter plus a `__meta__` JSON blob recording
the architecture, its constructor arguments, the training run, and the
held-out metrics — enough to rebuild and serve the model without the caller
knowing which architecture it is.

Checkpoints are not version-controlled. Regenerate them with:

    python scripts/train_behaviour.py --all --model lstm
    python scripts/train_behaviour.py --all --model transformer
