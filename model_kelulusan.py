# ============================================
# Tugas Pertemuan 7 - Langkah 6
# Eksperimen Model Kelulusan Mahasiswa
# ============================================

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------
# 1️⃣ Setup
# --------------------------------------------
tf.random.set_seed(42)
CSV_PATH = "processed_kelulusan.csv"
RANDOM_STATE = 42

# --------------------------------------------
# 2️⃣ Load dan preprocessing ulang
# --------------------------------------------
df = pd.read_csv(CSV_PATH)
target_col = "Lulus"
X = df.drop(columns=[target_col])
y = df[target_col]

# Encode dan scale
X = pd.get_dummies(X, drop_first=True)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE
)

print("Data train:", X_train.shape)
print("Data validasi:", X_val.shape)
print("Data uji:", X_test.shape)

# --------------------------------------------
# 3️⃣ Fungsi pembuat model
# --------------------------------------------
def build_model(neurons=32, optimizer="adam", l2_reg=0.001, dropout_rate=0.2):
    model = keras.Sequential([
        layers.Input(shape=(X_train.shape[1],)),
        layers.Dense(neurons, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)),
        layers.Dropout(dropout_rate),
        layers.Dense(neurons//2, activation="relu"),
        layers.Dense(1, activation="sigmoid")
    ])
    
    if optimizer == "adam":
        opt = keras.optimizers.Adam(learning_rate=0.001)
    else:
        opt = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
    
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model

# --------------------------------------------
# 4️⃣ Eksperimen beberapa kombinasi
# --------------------------------------------
configurations = [
    {"neurons": 32, "optimizer": "adam"},
    {"neurons": 64, "optimizer": "adam"},
    {"neurons": 128, "optimizer": "adam"},
    {"neurons": 64, "optimizer": "sgd"}
]

results = []

for cfg in configurations:
    print(f"\n🚀 Training model: neurons={cfg['neurons']} | optimizer={cfg['optimizer']}")
    model = build_model(neurons=cfg["neurons"], optimizer=cfg["optimizer"])
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=4,
        verbose=0
    )
    
    # Evaluasi di test set
    y_pred_prob = model.predict(X_test).ravel()
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_pred_prob)
    
    results.append({
        "neurons": cfg["neurons"],
        "optimizer": cfg["optimizer"],
        "accuracy": acc,
        "f1": f1,
        "auc": auc
    })
    
    print(f"Akurasi={acc:.3f} | F1={f1:.3f} | AUC={auc:.3f}")
    
    # Simpan grafik learning curve
    plt.figure()
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title(f"Learning Curve ({cfg['neurons']} neuron, {cfg['optimizer']})")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig(f"learning_curve_{cfg['neurons']}_{cfg['optimizer']}.png")
    plt.close()

# --------------------------------------------
# 5️⃣ Hasil akhir dan evaluasi model terbaik
# --------------------------------------------
results_df = pd.DataFrame(results)
print("\n📊 Hasil Eksperimen:")
print(results_df)

best_model = results_df.loc[results_df["accuracy"].idxmax()]
print(f"\n🏆 Model terbaik: {best_model.to_dict()}")

# Latih ulang model terbaik di seluruh data
final_model = build_model(
    neurons=int(best_model["neurons"]),
    optimizer=best_model["optimizer"]
)
final_model.fit(X_train, y_train, epochs=100, verbose=0)

# Confusion Matrix & ROC Curve
y_pred_prob = final_model.predict(X_test).ravel()
y_pred = (y_pred_prob > 0.5).astype(int)
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig("confusion_matrix.png")
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
plt.figure()
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test, y_pred_prob):.3f}")
plt.plot([0,1], [0,1], "--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig("roc_curve.png")
plt.close()

print("\n✅ Semua eksperimen selesai. Grafik dan hasil tersimpan sebagai file PNG.")
