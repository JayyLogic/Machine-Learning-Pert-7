# ============================================
# Tugas Pertemuan 7
# Analisis Awal Data Kelulusan (Langkah 1–5)
# ============================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# --- Konfigurasi ---
CSV_PATH = "processed_kelulusan.csv"
RANDOM_STATE = 42

# --- 1️⃣ Load dataset ---
print("Memuat dataset:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
print("Shape:", df.shape)
print("Kolom:", list(df.columns))

print("\nContoh data:")
print(df.head())

# --- 2️⃣ Identifikasi target dan fitur ---
possible_targets = ["Lulus", "status_kelulusan", "Lulus?", "lulus"]
target_col = None
for t in possible_targets:
    if t in df.columns:
        target_col = t
        break
if target_col is None:
    target_col = df.columns[-1]
    print(f"[INFO] Target tidak ditemukan otomatis. Gunakan kolom terakhir: {target_col}")

print(f"\nKolom target yang digunakan: '{target_col}'")
X = df.drop(columns=[target_col])
y = df[target_col]

# Ubah label ke 0/1
if y.dtype == object or str(y.dtype).startswith("category"):
    y = y.map({
        "Lulus": 1, "lulus": 1, "Ya": 1, "ya": 1, "Passed": 1, "passed": 1,
        "Tidak": 0, "tidak": 0, "Tidak Lulus": 0, "Failed": 0
    }).astype(int)
y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)

print("\nDistribusi label:")
print(y.value_counts())

# --- 3️⃣ Preprocessing ---
# Hilangkan kolom ID kalau ada
drop_cols = [c for c in X.columns if X[c].dtype == object and X[c].nunique() > 50]
if drop_cols:
    print("\nMenghapus kolom yang diduga sebagai ID:", drop_cols)
    X = X.drop(columns=drop_cols)

X = pd.get_dummies(X, drop_first=True)
print("\nBentuk fitur setelah encoding:", X.shape)

# Normalisasi data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 4️⃣ Split dataset ---
print("\nMembagi dataset...")
min_class_count = y.value_counts().min()
if min_class_count < 2 or len(y) < 8:
    print("[INFO] Dataset kecil. Split tanpa stratify untuk menghindari error.")
    stratify_opt = None
else:
    stratify_opt = y

X_train, X_temp, y_train, y_temp = train_test_split(
    X_scaled, y, test_size=0.3, stratify=stratify_opt, random_state=RANDOM_STATE
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, stratify=None, random_state=RANDOM_STATE
)

print("Data train:", X_train.shape)
print("Data validasi:", X_val.shape)
print("Data uji:", X_test.shape)

# --- 5️⃣ Ringkasan akhir ---
print("\n=== Selesai ===")
print(f"Total data: {len(df)} | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
print("Siap lanjut ke langkah 6 (pembuatan model).")
