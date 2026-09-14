# Fault Detection - Hydraulic System

Deep learning model comparison for fault detection on hydraulic system sensor data, comparing CNN, LSTM, CNN-LSTM, MLP, and RNN architectures.

## Dataset

`dataset3_hydraulic_system.csv` — 200,000 rows, 17 sensor features (pressure, motor power, flow, temperature, vibration, efficiency, cooling), and a binary `fault` label (0 = normal, 1 = fault). The classes are imbalanced at roughly 26:1 (normal:fault).

## Approach

- Chronological train/validation/test split (70/15/15), no shuffling, to avoid time-series leakage.
- Features standardized with a scaler fit only on the training split.
- For sequential models (CNN, LSTM, CNN-LSTM, RNN): data is reshaped into sliding windows of 16 consecutive rows, labeled using the center row's class.
- For the MLP: raw rows are used directly (no windowing).
- SMOTE oversampling applied to the training data only, plus class weighting during training.
- Per-model classification threshold tuned on the validation set to maximize F1 score.

## Models

| Model | Input | Architecture |
|---|---|---|
| CNN | windowed (16, 17) | 3x Conv1D -> MaxPool -> Dense |
| LSTM | windowed (16, 17) | 2x stacked LSTM -> Dense |
| CNN-LSTM | windowed (16, 17) | Conv1D -> MaxPool -> LSTM -> Dense |
| MLP | raw rows (17,) | 4x Dense with dropout |
| RNN | windowed (16, 17) | 2x stacked SimpleRNN -> Dense |

## Files

- `fault_detection_models.py` — standalone script version
- `fault_detection_models.ipynb` — Colab-ready notebook version
- `dataset3_hydraulic_system.csv` — dataset

## Running it

### Colab
Upload `fault_detection_models.ipynb` to Colab, run the cells top to bottom, and upload `dataset3_hydraulic_system.csv` when prompted.

### Locally
```
pip install tensorflow imbalanced-learn scikit-learn pandas numpy
python fault_detection_models.py
```
Make sure `dataset3_hydraulic_system.csv` is in the same directory.

## Output

Running either version prints per-model Accuracy, Precision, Recall, and F1-score, and saves a summary to `final_model_results.csv`.
