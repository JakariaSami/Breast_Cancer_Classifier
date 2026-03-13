# Breast Cancer Classifier

A neural network built with TensorFlow/Keras to classify breast tumors as **malignant** or **benign**, using the Wisconsin Breast Cancer dataset.

---

## Results

| Metric              | Score      |
| ------------------- | ---------- |
| Test Accuracy       | **99.12%** |
| Test Loss           | 0.085      |
| Malignant Precision | 1.00       |
| Benign Recall       | 1.00       |

---

## Dataset

The [Wisconsin Breast Cancer dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html) is built into scikit-learn — no download required.

- **569 samples**, **30 features** (radius, texture, perimeter, area, etc. of cell nuclei)
- **Binary target:** 0 = Malignant, 1 = Benign
- **Split:** 60% train / 20% validation / 20% test

---

## Model Architecture

### Baseline — Logistic Regression

A single neuron with a sigmoid activation (no hidden layers).

```
Input (30) → Dense(1, sigmoid)
```

### Final Model

A 3-layer neural network with L2 regularization and Dropout.

```
Input (30)
  → Dense(64, ReLU) + L2(0.001) + Dropout(0.3)
  → Dense(32, ReLU) + L2(0.001) + Dropout(0.2)
  → Dense(16, ReLU)
  → Dense(1, sigmoid)
```

**Training config:**

- Optimizer: Adam (lr = 0.001)
- Loss: Binary Crossentropy
- Early Stopping: patience = 25, restores best weights

---

## Concepts Implemented

| Concept                        | Where Applied                 |
| ------------------------------ | ----------------------------- |
| Logistic / Sigmoid function    | Baseline model output layer   |
| Normalization (StandardScaler) | Applied to all input features |
| Gradient Descent               | Via Adam optimizer            |
| ReLU activation                | All hidden layers             |
| Multiple hidden layers         | Final model architecture      |
| L2 Regularization              | Hidden layers (λ = 0.001)     |
| Dropout                        | After first two hidden layers |
| Early Stopping                 | Training callback             |
| Train / Val / Test split       | Data pipeline                 |
| Confusion Matrix               | Final evaluation              |

---

## How to Run

### Option 1 — Google Colab (recommended, no setup)

Upload `Classifier.ipynb` to [colab.research.google.com](https://colab.research.google.com) and run all cells.

### Option 2 — Local

```bash
pip install tensorflow scikit-learn matplotlib seaborn numpy
jupyter notebook Classifier.ipynb
```

---

## Dependencies

| Library            | Purpose                                                          |
| ------------------ | ---------------------------------------------------------------- |
| TensorFlow / Keras | Building and training the neural network                         |
| scikit-learn       | Dataset, train/val/test split, normalization, evaluation metrics |
| NumPy              | Array operations                                                 |
| Matplotlib         | Plotting                                                         |
| Seaborn            | Confusion matrix heatmap                                         |
