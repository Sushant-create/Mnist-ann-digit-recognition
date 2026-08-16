"""
================================================================================
 Handwritten Digit Recognition using Artificial Neural Networks (ANN)
================================================================================
 Dataset : MNIST Handwritten Digits (CSV format)
           https://www.kaggle.com/datasets/oddrationale/mnist-in-csv
 Author  : Sushant Tiwari
 License : All Rights Reserved (see LICENSE)

 This script performs the full pipeline required for the assignment:
   Task 1 - Data Understanding
   Task 2 - Data Preprocessing
   Task 3 - Model Development (ANN)
   Task 4 - Model Evaluation

 HOW TO RUN
 ----------
 1. Download `mnist_train.csv` and `mnist_test.csv` from the Kaggle link
    above and place them inside the `data/` folder.
 2. Install dependencies:  pip install -r requirements.txt
 3. Run:                   python main.py
    (or simply run `python run.py` / `run.bat` / `./run.sh`)

 All plots and reports are saved inside `outputs/`.
 The trained model is saved inside `saved_model/`.
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # safe for headless / script execution
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
DATA_DIR = "data"
OUTPUT_DIR = "outputs"
MODEL_DIR = "saved_model"
TRAIN_CSV = os.path.join(DATA_DIR, "mnist_train.csv")
TEST_CSV = os.path.join(DATA_DIR, "mnist_test.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

RANDOM_STATE = 42


# ==========================================================================
# TASK 1: DATA UNDERSTANDING
# ==========================================================================
def load_data():
    """Load the MNIST CSV dataset(s) using Pandas.

    The Kaggle dataset ships as two files (mnist_train.csv / mnist_test.csv),
    each with columns: label, pixel0, pixel1, ..., pixel783.
    Per the assignment we combine both files and later perform our own
    80/20 train-test split (Task 2).
    """
    frames = []
    if os.path.exists(TRAIN_CSV):
        frames.append(pd.read_csv(TRAIN_CSV))
    if os.path.exists(TEST_CSV):
        frames.append(pd.read_csv(TEST_CSV))

    if not frames:
        raise FileNotFoundError(
            "Could not find mnist_train.csv / mnist_test.csv in the 'data/' folder.\n"
            "Download the dataset from:\n"
            "https://www.kaggle.com/datasets/oddrationale/mnist-in-csv\n"
            "and place the CSV files inside the 'data' directory."
        )

    df = pd.concat(frames, ignore_index=True)
    return df


def explore_data(df):
    print("\n===== TASK 1: DATA UNDERSTANDING =====")

    print("\nFirst five records:")
    print(df.head())

    print("\nDataset dimensions (rows, columns):", df.shape)

    print("\nSummary information:")
    print(df.info())

    print("\nInput features: pixel0 ... pixel783  (784 grayscale pixel intensity "
          "columns, each in the range 0-255, representing a flattened 28x28 image)")
    print("Target variable: 'label' column (the digit 0-9 the image represents)")

    # Display one sample handwritten digit
    sample_row = df.iloc[0]
    sample_label = sample_row["label"]
    sample_image = sample_row.drop("label").values.reshape(28, 28).astype("uint8")

    plt.figure(figsize=(3, 3))
    plt.imshow(sample_image, cmap="gray")
    plt.title(f"Sample Digit - Label: {sample_label}")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "sample_digit.png"))
    plt.close()
    print(f"\nSaved sample digit image -> {OUTPUT_DIR}/sample_digit.png")


# ==========================================================================
# TASK 2: DATA PREPROCESSING
# ==========================================================================
def preprocess_data(df):
    print("\n===== TASK 2: DATA PREPROCESSING =====")

    # Check for missing values
    missing_total = df.isnull().sum().sum()
    print(f"\nTotal missing values in dataset: {missing_total}")

    # Separate features (X) and target (y)
    X = df.drop("label", axis=1).values
    y = df["label"].values

    # Normalize pixel values to the range 0-1
    X = X.astype("float32") / 255.0

    # Split into 80% training and 20% testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training samples: {X_train.shape[0]} | Testing samples: {X_test.shape[0]}")

    # One-hot encode the labels
    y_train_cat = to_categorical(y_train, num_classes=10)
    y_test_cat = to_categorical(y_test, num_classes=10)

    print("Labels one-hot encoded to shape:", y_train_cat.shape)

    return X_train, X_test, y_train, y_test, y_train_cat, y_test_cat


# ==========================================================================
# TASK 3: MODEL DEVELOPMENT
# ==========================================================================
def build_model():
    print("\n===== TASK 3: MODEL DEVELOPMENT =====")

    model = keras.Sequential([
        layers.Input(shape=(784,), name="Input_Layer"),
        layers.Dense(128, activation="relu", name="Hidden_Layer_1"),
        layers.Dense(64, activation="relu", name="Hidden_Layer_2"),
        layers.Dense(10, activation="softmax", name="Output_Layer"),
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()
    return model


def train_model(model, X_train, y_train_cat):
    history = model.fit(
        X_train, y_train_cat,
        epochs=10,
        batch_size=128,
        validation_split=0.1,
        verbose=2,
    )
    return history


# ==========================================================================
# TASK 4: MODEL EVALUATION
# ==========================================================================
def evaluate_model(model, history, X_test, y_test, y_test_cat):
    print("\n===== TASK 4: MODEL EVALUATION =====")

    # Predictions
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Test accuracy
    test_loss, test_accuracy = model.evaluate(X_test, y_test_cat, verbose=0)
    print(f"\nTest Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=range(10), yticklabels=range(10))
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"))
    plt.close()
    print(f"Saved confusion matrix -> {OUTPUT_DIR}/confusion_matrix.png")

    # Classification Report
    report = classification_report(y_test, y_pred, digits=4)
    print("\nClassification Report:\n", report)
    with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
        f.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n\n")
        f.write(report)
    print(f"Saved classification report -> {OUTPUT_DIR}/classification_report.txt")

    # Accuracy vs Epoch graph
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["accuracy"], label="Training Accuracy", marker="o")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy", marker="o")
    plt.title("Accuracy vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "accuracy_vs_epoch.png"))
    plt.close()

    # Loss vs Epoch graph
    plt.figure(figsize=(7, 5))
    plt.plot(history.history["loss"], label="Training Loss", marker="o")
    plt.plot(history.history["val_loss"], label="Validation Loss", marker="o")
    plt.title("Loss vs Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "loss_vs_epoch.png"))
    plt.close()

    print(f"Saved accuracy/loss graphs -> {OUTPUT_DIR}/")

    return y_pred, test_accuracy


def main():
    df = load_data()
    explore_data(df)

    X_train, X_test, y_train, y_test, y_train_cat, y_test_cat = preprocess_data(df)

    model = build_model()
    history = train_model(model, X_train, y_train_cat)

    evaluate_model(model, history, X_test, y_test, y_test_cat)

    model_path = os.path.join(MODEL_DIR, "mnist_ann_model.keras")
    model.save(model_path)
    print(f"\nModel saved to -> {model_path}")

    print("\n===== OBSERVATIONS (fill in / verify after your run) =====")
    print("1. The ANN converges within a few epochs, with training accuracy "
          "rising faster than validation accuracy, indicating quick learning "
          "on the relatively simple, well-normalized MNIST pixel data.")
    print("2. Most misclassifications in the confusion matrix occur between "
          "visually similar digit pairs (e.g., 4/9, 3/5, 7/1), which is "
          "expected since a plain ANN ignores 2D spatial structure.")
    print("3. Per-class precision/recall in the classification report is "
          "fairly uniform, showing the model does not have a first strong "
          "class bias despite MNIST's class distribution being close to balanced.")
    print("4. Validation loss plateauing (or slightly rising) in later epochs "
          "while training loss keeps falling would indicate the onset of "
          "overfitting - worth watching if you extend training beyond 10 epochs.")

    print("\nDone. Check the 'outputs/' folder for all saved plots and reports.")


if __name__ == "__main__":
    main()
