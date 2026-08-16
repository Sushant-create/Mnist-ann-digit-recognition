"""
================================================================================
 test_samples.py
--------------------------------------------------------------------------------
 Loads the trained ANN model (saved_model/mnist_ann_model.h5) and runs it on a
 handful of random handwritten digit samples from the test set, displaying the
 predicted label next to the true label for a quick visual sanity check.

 Author  : Sushant Tiwari
 License : All Rights Reserved (see LICENSE)

 HOW TO RUN
 ----------
     python test_samples.py
 (Run main.py first so that a trained model exists in 'saved_model/'.)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorflow import keras

DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_PATH = os.path.join("saved_model", "mnist_ann_model.keras")
TRAIN_CSV = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "mnist_test.csv")

NUM_SAMPLES = 10
RANDOM_STATE = 7


def load_any_available_csv():
    """Prefer the dedicated test CSV; fall back to the train CSV if needed."""
    if os.path.exists(TEST_CSV):
        return pd.read_csv(TEST_CSV)
    if os.path.exists(TRAIN_CSV):
        return pd.read_csv(TRAIN_CSV)
    raise FileNotFoundError(
        "No dataset CSV found in 'data/'. Download the MNIST CSV dataset from:\n"
        "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv"
    )


def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"No trained model found at '{MODEL_PATH}'.\n"
            "Run 'python main.py' first to train and save the model."
        )

    print("Loading trained model...")
    model = keras.models.load_model(MODEL_PATH)

    print("Loading sample data...")
    df = load_any_available_csv()

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx = rng.choice(len(df), size=NUM_SAMPLES, replace=False)
    samples = df.iloc[sample_idx]

    X_samples = samples.drop("label", axis=1).values.astype("float32") / 255.0
    y_true = samples["label"].values

    predictions = model.predict(X_samples)
    y_pred = np.argmax(predictions, axis=1)
    confidences = np.max(predictions, axis=1)

    # Plot a grid of samples with predicted vs actual labels
    cols = 5
    rows = int(np.ceil(NUM_SAMPLES / cols))
    plt.figure(figsize=(cols * 2.2, rows * 2.6))

    correct_count = 0
    for i in range(NUM_SAMPLES):
        img = X_samples[i].reshape(28, 28)
        is_correct = y_true[i] == y_pred[i]
        correct_count += int(is_correct)

        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap="gray")
        color = "green" if is_correct else "red"
        plt.title(
            f"True: {y_true[i]} | Pred: {y_pred[i]}\n({confidences[i]*100:.1f}%)",
            color=color, fontsize=9
        )
        plt.axis("off")

    plt.suptitle(f"Sample Predictions ({correct_count}/{NUM_SAMPLES} correct)")
    plt.tight_layout()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "test_predictions.png")
    plt.savefig(out_path)
    plt.close()

    print(f"\n{correct_count}/{NUM_SAMPLES} sample predictions correct.")
    print(f"Saved visualization -> {out_path}")


if __name__ == "__main__":
    main()
