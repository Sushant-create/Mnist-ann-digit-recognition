# Handwritten Digit Recognition using Artificial Neural Networks (ANN)

A postal service organization wants to automate the recognition of handwritten
digits on postal codes. This project builds and evaluates an Artificial Neural
Network (ANN) that classifies handwritten digits (0-9) from the MNIST dataset.

## Objective

To develop, train, and evaluate an Artificial Neural Network capable of
accurately classifying handwritten digit images (0-9), demonstrating a
practical automation solution for postal code digit recognition.

## Dataset Link

**MNIST Handwritten Digits Dataset (CSV format)**
https://www.kaggle.com/datasets/oddrationale/mnist-in-csv

The dataset contains `mnist_train.csv` and `mnist_test.csv`, each row holding
a `label` column (digit 0-9) followed by 784 `pixel0`...`pixel783` columns —
the flattened grayscale pixel intensities (0-255) of a 28x28 handwritten digit
image.

> Download both CSV files from the link above and place them inside the
> `data/` folder before running the project (see **Project Structure** below).

## Libraries Used

| Library | Purpose |
|---|---|
| pandas | Loading and exploring the CSV dataset |
| numpy | Numerical array operations |
| matplotlib | Plotting sample digits and training curves |
| seaborn | Confusion matrix heatmap |
| scikit-learn | Train/test split, confusion matrix, classification report |
| tensorflow / keras | Building, training, and saving the ANN |

## Methodology

1. **Data Understanding** — Load the dataset with Pandas, inspect its shape,
   structure, and a sample digit image.
2. **Data Preprocessing** — Check for missing values, separate features (`X`)
   and target (`y`), normalize pixel values to `[0, 1]`, split the data 80/20
   into training and testing sets, and one-hot encode the digit labels.
3. **Model Development** — Build a fully-connected ANN in Keras and train it
   for 10 epochs using the Adam optimizer and categorical crossentropy loss.
4. **Model Evaluation** — Measure test accuracy, generate a confusion matrix
   and classification report, and plot accuracy/loss curves across epochs.
5. **Testing on Samples** — Run the trained model on a handful of individual
   digit images to visually confirm predictions.

## Model Architecture

```
Input Layer          : 784 neurons (flattened 28x28 pixel image)
Hidden Layer 1        : 128 neurons, ReLU activation
Hidden Layer 2        : 64 neurons, ReLU activation
Output Layer          : 10 neurons, Softmax activation (digits 0-9)

Optimizer  : Adam
Loss       : Categorical Crossentropy
Metric     : Accuracy
Epochs     : 10
Batch Size : 128
```

## Results

Running `main.py` (or `index.ipynb`) on the full MNIST dataset produces, inside
`outputs/`:

- `sample_digit.png` — an example handwritten digit from the dataset
- `accuracy_vs_epoch.png` — training vs. validation accuracy across epochs
- `loss_vs_epoch.png` — training vs. validation loss across epochs
- `confusion_matrix.png` — a heatmap of predicted vs. true digit labels
- `classification_report.txt` — precision, recall, and F1-score per digit,
  plus overall test accuracy
- `test_predictions.png` (from `test_samples.py`) — a grid of sample test
  images with predicted vs. actual labels

**Observations** (verify exact figures after your own run — they depend on
your data and hardware):

1. The ANN converges quickly on MNIST, with training accuracy climbing faster
   than validation accuracy across the 10 epochs — expected for a relatively
   simple, well-normalized dataset.
2. Misclassifications concentrate on visually similar digit pairs (e.g. 4/9,
   3/5, 7/1), since a plain ANN treats pixels independently and ignores the
   2D spatial structure a CNN would exploit.
3. Precision and recall stay fairly uniform across the ten digit classes,
   showing no strong class-level bias.
4. If validation loss plateaus or rises while training loss keeps falling in
   later epochs, that signals early overfitting — worth watching if training
   is extended beyond 10 epochs.

## Conclusion

This project showed that a simple Artificial Neural Network with two hidden
layers can classify handwritten digits from the MNIST dataset with high
accuracy, making it a practical approach for automating postal code digit
recognition. The hidden layers are central to this: each one learns
progressively more abstract combinations of pixel intensities, allowing the
network to separate digit classes that are not linearly separable in raw
pixel space — without them, the model would collapse into a single linear
classifier. Compared to traditional Machine Learning methods (e.g. SVM,
Random Forest), Deep Learning models like this ANN learn useful feature
representations directly from raw data, removing the need for hand-crafted
feature engineering. A key limitation of a plain ANN, however, is that
flattening the image discards spatial relationships between neighboring
pixels, capping its accuracy and robustness — for instance to shifted or
rotated digits — compared to a Convolutional Neural Network purpose-built
for image data.

## Project Structure

```
mnist-ann-digit-recognition/
├── data/                       # Place mnist_train.csv & mnist_test.csv here
├── outputs/                    # Generated plots and reports (auto-created)
├── saved_model/                # Trained model saved here (auto-created)
├── index.ipynb                 # Full walkthrough notebook (Tasks 1-5)
├── main.py                     # End-to-end training/evaluation script
├── test_samples.py             # Visual sanity check on sample predictions
├── run.py                      # Direct launch file (installs deps + runs all)
├── run.sh                      # Linux/macOS launch wrapper
├── run.bat                     # Windows launch wrapper
├── requirements.txt            # Python dependencies
├── .gitignore
├── .gitattributes
├── LICENSE                     # Copyright / All Rights Reserved
└── README.md
```

## How to Run

1. Download `mnist_train.csv` and `mnist_test.csv` from the
   [Kaggle dataset](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
   and place them inside the `data/` folder.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run everything with the direct launch file:
   ```
   python run.py
   ```
   (or `./run.sh` on Linux/macOS, `run.bat` on Windows)

   Alternatively, run steps individually:
   ```
   python main.py            # Tasks 1-4: load, preprocess, train, evaluate
   python test_samples.py    # Visual test on sample predictions
   ```
   Or open `index.ipynb` in Jupyter/Colab to run the full notebook version.

---


