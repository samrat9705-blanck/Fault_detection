import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense, Conv1D, MaxPooling1D, Flatten,
    LSTM, SimpleRNN, Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_f1 = 0.5, 0.0
    for t in np.arange(0.05, 0.95, 0.01):
        preds = (y_prob >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, float(t)
    return best_t


print("=" * 65)
print("  STEP 1 - Loading Hydraulic System Dataset")
print("=" * 65)

df = pd.read_csv("dataset3_hydraulic_system.csv")

TARGET = "fault"
FEATURE_COLS = [c for c in df.columns if c != TARGET]

X_raw = df[FEATURE_COLS].values
y_raw = df[TARGET].values

print(f"Dataset shape   : {df.shape}")
print(f"Features        : {len(FEATURE_COLS)}")
print(f"Class dist      : Normal={np.sum(y_raw==0):,}  Fault={np.sum(y_raw==1):,}")
print(f"Imbalance ratio : {np.sum(y_raw==0)/np.sum(y_raw==1):.1f} : 1")


print("\n" + "=" * 65)
print("  STEP 2 - Chronological Split + Standardise")
print("=" * 65)

X_temp_r, X_test_r, y_temp_r, y_test_r = train_test_split(
    X_raw, y_raw, test_size=0.15, shuffle=False
)
X_train_r, X_val_r, y_train_r, y_val_r = train_test_split(
    X_temp_r, y_temp_r, test_size=0.1765, shuffle=False
)

scaler = StandardScaler().fit(X_train_r)
X_train_r = scaler.transform(X_train_r)
X_val_r = scaler.transform(X_val_r)
X_test_r = scaler.transform(X_test_r)

print("Row-level splits (MLP):")
print(f"  Train : {X_train_r.shape}  fault={np.sum(y_train_r==1):,}")
print(f"  Val   : {X_val_r.shape}    fault={np.sum(y_val_r==1):,}")
print(f"  Test  : {X_test_r.shape}   fault={np.sum(y_test_r==1):,}")

WINDOW_SIZE = 16
HALF = WINDOW_SIZE // 2

X_all_scaled = scaler.transform(X_raw)

print(f"\nBuilding sliding windows  (ws={WINDOW_SIZE}, center-row label) ...")
wins_X, wins_y = [], []
for i in range(HALF, len(X_all_scaled) - HALF):
    wins_X.append(X_all_scaled[i - HALF: i + HALF])
    wins_y.append(y_raw[i])

wins_X = np.array(wins_X, dtype=np.float32)
wins_y = np.array(wins_y, dtype=np.int32)

print(f"  Total windows : {len(wins_X):,}")
print(f"  Class dist    : Normal={np.sum(wins_y==0):,}  Fault={np.sum(wins_y==1):,}")
print(f"  Imbalance     : {np.sum(wins_y==0)/np.sum(wins_y==1):.1f} : 1")

X_temp_w, X_test_w, y_temp_w, y_test_w = train_test_split(
    wins_X, wins_y, test_size=0.15, shuffle=False
)
X_train_w, X_val_w, y_train_w, y_val_w = train_test_split(
    X_temp_w, y_temp_w, test_size=0.1765, shuffle=False
)

print("\nWindow-level splits (CNN / LSTM / CNN-LSTM / RNN):")
print(f"  Train : {X_train_w.shape}  fault={np.sum(y_train_w==1):,}")
print(f"  Val   : {X_val_w.shape}    fault={np.sum(y_val_w==1):,}")
print(f"  Test  : {X_test_w.shape}   fault={np.sum(y_test_w==1):,}")


print("\n" + "=" * 65)
print("  STEP 3 - SMOTE Oversampling (train only)")
print("=" * 65)

smote = SMOTE(random_state=SEED)

X_train_mlp, y_train_mlp = smote.fit_resample(X_train_r, y_train_r)
print(f"MLP  after SMOTE : {X_train_mlp.shape}  "
      f"Normal={np.sum(y_train_mlp==0):,}  Fault={np.sum(y_train_mlp==1):,}")

n_tr_w, n_steps, n_feats = X_train_w.shape
X_tr_w_flat = X_train_w.reshape(n_tr_w, n_steps * n_feats)

smote2 = SMOTE(random_state=SEED)
X_tr_w_bal_flat, y_train_seq = smote2.fit_resample(X_tr_w_flat, y_train_w)
X_train_seq = X_tr_w_bal_flat.reshape(-1, n_steps, n_feats)

print(f"Seq  after SMOTE : {X_train_seq.shape}  "
      f"Normal={np.sum(y_train_seq==0):,}  Fault={np.sum(y_train_seq==1):,}")


def class_weights(y):
    cls = np.unique(y)
    wts = compute_class_weight("balanced", classes=cls, y=y)
    return dict(zip(cls.tolist(), wts.tolist()))


cw_mlp = class_weights(y_train_mlp)
cw_seq = class_weights(y_train_seq)


INPUT_SEQ = (n_steps, n_feats)
INPUT_MLP = (n_feats,)


def build_cnn():
    m = Sequential(name="CNN")
    m.add(Conv1D(32, kernel_size=3, activation="relu", input_shape=INPUT_SEQ,
                 padding="same"))
    m.add(Conv1D(64, kernel_size=3, activation="relu", padding="same"))
    m.add(Conv1D(128, kernel_size=3, activation="relu", padding="same"))
    m.add(MaxPooling1D(pool_size=2))
    m.add(Flatten())
    m.add(Dense(256, activation="relu"))
    m.add(Dropout(0.5))
    m.add(Dense(128, activation="relu"))
    m.add(Dropout(0.5))
    m.add(Dense(1, activation="sigmoid"))
    return m


def build_lstm():
    m = Sequential(name="LSTM")
    m.add(LSTM(128, return_sequences=True, activation="tanh",
               recurrent_dropout=0.2, input_shape=INPUT_SEQ))
    m.add(Dropout(0.2))
    m.add(LSTM(128, activation="tanh", recurrent_dropout=0.2))
    m.add(Dropout(0.2))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    return m


def build_cnn_lstm():
    m = Sequential(name="CNN-LSTM")
    m.add(Conv1D(128, kernel_size=3, activation="relu",
                 input_shape=INPUT_SEQ, padding="same"))
    m.add(MaxPooling1D(pool_size=2))
    m.add(LSTM(128, activation="tanh", recurrent_dropout=0.2))
    m.add(Dropout(0.2))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    return m


def build_mlp():
    m = Sequential(name="MLP")
    m.add(Dense(512, activation="relu", input_shape=INPUT_MLP))
    m.add(Dropout(0.4))
    m.add(Dense(256, activation="relu"))
    m.add(Dropout(0.3))
    m.add(Dense(128, activation="relu"))
    m.add(Dropout(0.2))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    return m


def build_rnn():
    m = Sequential(name="RNN")
    m.add(SimpleRNN(128, return_sequences=True, activation="tanh",
                     input_shape=INPUT_SEQ))
    m.add(Dropout(0.2))
    m.add(SimpleRNN(64, activation="tanh"))
    m.add(Dropout(0.2))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    return m


EPOCHS = 50
BATCH_SIZE = 128


def make_callbacks():
    return [
        EarlyStopping(
            monitor="val_loss", patience=20,
            restore_best_weights=True, verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.1,
            patience=10, min_lr=1e-6, verbose=1,
        ),
    ]


registry = [
    ("CNN", build_cnn, X_train_seq, y_train_seq, X_val_w, y_val_w, X_test_w, y_test_w, cw_seq),
    ("LSTM", build_lstm, X_train_seq, y_train_seq, X_val_w, y_val_w, X_test_w, y_test_w, cw_seq),
    ("CNN-LSTM", build_cnn_lstm, X_train_seq, y_train_seq, X_val_w, y_val_w, X_test_w, y_test_w, cw_seq),
    ("MLP", build_mlp, X_train_mlp, y_train_mlp, X_val_r, y_val_r, X_test_r, y_test_r, cw_mlp),
    ("RNN", build_rnn, X_train_seq, y_train_seq, X_val_w, y_val_w, X_test_w, y_test_w, cw_seq),
]


print("\n" + "=" * 65)
print("  STEP 4 - Training & Evaluation")
print("=" * 65)

results = []

for name, build_fn, X_tr, y_tr, X_v, y_v, X_te, y_te, cw in registry:
    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Train {X_tr.shape} | Val {X_v.shape} | Test {X_te.shape}")

    model = build_fn()
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    model.fit(
        X_tr, y_tr,
        validation_data=(X_v, y_v),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=cw,
        callbacks=make_callbacks(),
        verbose=1,
    )

    val_prob = model.predict(X_v, batch_size=BATCH_SIZE, verbose=0).ravel()
    opt_t = find_best_threshold(y_v, val_prob)
    print(f"\n  Optimal threshold (val F1): {opt_t:.2f}")

    test_prob = model.predict(X_te, batch_size=BATCH_SIZE, verbose=0).ravel()
    y_pred = (test_prob >= opt_t).astype(int)

    acc = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, zero_division=0)
    rec = recall_score(y_te, y_pred, zero_division=0)
    f1 = f1_score(y_te, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": round(acc * 100, 2),
        "Precision": round(prec * 100, 2),
        "Recall": round(rec * 100, 2),
        "F1-score": round(f1 * 100, 2),
    })

    print(f"\n  {name} -> Acc={acc*100:.2f}%  Prec={prec*100:.2f}%  "
          f"Rec={rec*100:.2f}%  F1={f1*100:.2f}%\n")
    print(classification_report(y_te, y_pred, target_names=["Normal", "Fault"]))


results_df = pd.DataFrame(results)

print("\n" + "=" * 65)
print("  FINAL RESULTS")
print("=" * 65)
print(results_df.to_string(index=False))

results_df.to_csv("final_model_results.csv", index=False)
print("\nResults saved to -> final_model_results.csv")
