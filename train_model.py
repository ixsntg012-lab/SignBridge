"""
Train Sign Language Model
==========================
Uses augmented dataset for better generalization.

Steps:
  1. python augment_data.py       ← generate augmented CSV
  2. python train_model.py        ← train + evaluate

Output:
  models/sign_model.pkl
  models/eval_report.txt
  models/confusion_matrix.png
"""

import pandas as pd
import numpy as np
import joblib, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────────────────────
AUG_CSV    = "data/signs_augmented.csv"
RAW_CSV    = "data/signs.csv"
MODEL_PATH = "models/sign_model.pkl"
REPORT_PATH= "models/eval_report.txt"
CM_PATH    = "models/confusion_matrix.png"

os.makedirs("models", exist_ok=True)

# Use augmented if available, else fallback to raw
csv_path = AUG_CSV if os.path.exists(AUG_CSV) else RAW_CSV
print(f"Using: {csv_path}")

# ── Load ───────────────────────────────────────────────────────────────────
df = pd.read_csv(csv_path)
print(f"Total samples: {len(df)}")
print(f"Classes: {sorted(df['label'].unique())}")
print(f"Samples per class:\n{df['label'].value_counts().sort_index().to_string()}\n")

X_raw = df.drop("label", axis=1).values
y     = df["label"].values

# ── Feature extraction ─────────────────────────────────────────────────────
def extract(rows):
    out = []
    for row in rows:
        pts = row.reshape(21,3).copy()
        pts -= pts[0]                              # wrist-relative
        d = np.max(np.linalg.norm(pts, axis=1))
        if d > 0: pts /= d
        out.append(pts.flatten())
    return np.array(out, dtype=np.float32)

print("Extracting features...")
X = extract(X_raw)

# ── Split — stratified ─────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

# ── Model — Voting ensemble of two RF variants ─────────────────────────────
# Two complementary Random Forests (different hyperparams) → majority vote
# More robust than single model
rf1 = RandomForestClassifier(
    n_estimators=300, max_depth=None,
    min_samples_leaf=1, max_features="sqrt",
    random_state=42, n_jobs=-1)

rf2 = RandomForestClassifier(
    n_estimators=200, max_depth=30,
    min_samples_leaf=2, max_features="log2",
    random_state=7, n_jobs=-1)

model = VotingClassifier(
    estimators=[("rf1", rf1), ("rf2", rf2)],
    voting="soft",      # average probabilities → smoother confidence scores
    n_jobs=-1)

print("Training ensemble model...")
model.fit(X_train, y_train)

# ── Evaluate ───────────────────────────────────────────────────────────────
y_pred = model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=3)

print(f"\nTest Accuracy : {acc*100:.2f}%")
print(report)

# 5-fold stratified CV on full dataset
print("Running 5-fold cross-validation...")
skf      = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores= cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)
print(f"CV Accuracy   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# ── Save report ────────────────────────────────────────────────────────────
with open(REPORT_PATH, "w") as f:
    f.write("=== ASL Sign Language Recognition — Model Evaluation ===\n\n")
    f.write(f"Dataset       : {csv_path}\n")
    f.write(f"Total samples : {len(df)}\n")
    f.write(f"Train / Test  : {len(X_train)} / {len(X_test)}\n")
    f.write(f"Model         : Soft Voting Ensemble (RF x2)\n\n")
    f.write(f"Test Accuracy : {acc*100:.2f}%\n")
    f.write(f"5-Fold CV     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%\n\n")
    f.write("Per-Class Report:\n")
    f.write(report)
print(f"\n[Saved] {REPORT_PATH}")

# ── Confusion matrix ───────────────────────────────────────────────────────
labels = sorted(set(y))
cm     = confusion_matrix(y_test, y_pred, labels=labels)

fig, ax = plt.subplots(figsize=(15,13))
disp = ConfusionMatrixDisplay(cm, display_labels=[l.upper() for l in labels])
disp.plot(ax=ax, colorbar=True, cmap="Blues", xticks_rotation=45)
ax.set_title("ASL Recognition — Confusion Matrix", fontsize=14, pad=15)
plt.tight_layout()
plt.savefig(CM_PATH, dpi=120)
plt.close()
print(f"[Saved] {CM_PATH}")

# ── Save model ─────────────────────────────────────────────────────────────
joblib.dump(model, MODEL_PATH)
print(f"[Saved] {MODEL_PATH}")
print(f"\n✓ Done! Test accuracy: {acc*100:.2f}%")