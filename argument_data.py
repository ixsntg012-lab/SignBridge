"""
Dataset Augmentation — Sign Language
======================================
Augments existing hand landmark data (signs.csv) without a camera.
Each original sample → 4 augmented variants:
  1. Gaussian noise          (hand tremor simulation)
  2. Horizontal flip         (left hand ↔ right hand)
  3. Scale variation         (hand closer/farther from camera)
  4. Small rotation          (wrist tilt)

Result: ~5x more training data → better generalization

Usage:
    python augment_data.py

Output:
    data/signs_augmented.csv   ← use this for training
"""

import pandas as pd
import numpy as np
import os

CSV_PATH = "data/signs.csv"
OUT_PATH = "data/signs_augmented.csv"

np.random.seed(42)

# ── Load ──────────────────────────────────────────────────────────────────
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
print(f"Original samples: {len(df)}")
print(f"Labels: {sorted(df['label'].unique())}\n")

X_raw = df.drop("label", axis=1).values   # shape (N, 63)
y     = df["label"].values

# ── Helpers ───────────────────────────────────────────────────────────────
def to_pts(row):
    """Flatten row (63,) → (21,3) array."""
    return row.reshape(21, 3).copy()

def from_pts(pts):
    """(21,3) → (63,) flat row."""
    return pts.flatten()

def add_noise(pts, sigma=0.008):
    """Simulate natural hand tremor."""
    return pts + np.random.normal(0, sigma, pts.shape)

def flip_horizontal(pts):
    """Mirror x-axis — simulates left-hand variant."""
    p = pts.copy()
    p[:, 0] = -p[:, 0]
    return p

def scale_variation(pts, low=0.85, high=1.15):
    """Random scale — hand closer or farther."""
    s = np.random.uniform(low, high)
    return pts * s

def rotate_2d(pts, max_deg=15):
    """Small wrist rotation in xy-plane."""
    angle = np.radians(np.random.uniform(-max_deg, max_deg))
    c, s  = np.cos(angle), np.sin(angle)
    R     = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]])
    return pts @ R.T

def normalize(pts):
    """Wrist-relative + scale norm (same as training)."""
    p = pts - pts[0]                          # wrist-relative
    d = np.max(np.linalg.norm(p, axis=1))
    if d > 0: p /= d
    return p

# ── Augment ───────────────────────────────────────────────────────────────
augmented_rows = []

for i, (row, label) in enumerate(zip(X_raw, y)):
    pts = to_pts(row)

    # Original (keep as-is)
    augmented_rows.append((from_pts(pts), label))

    # 1. Noise (x3 — different random seeds each time)
    for _ in range(3):
        augmented_rows.append((from_pts(add_noise(pts)), label))

    # 2. Horizontal flip
    augmented_rows.append((from_pts(flip_horizontal(pts)), label))

    # 3. Flip + noise
    augmented_rows.append((from_pts(add_noise(flip_horizontal(pts))), label))

    # 4. Scale variation (x2)
    for _ in range(2):
        augmented_rows.append((from_pts(scale_variation(pts)), label))

    # 5. Rotation (x2)
    for _ in range(2):
        augmented_rows.append((from_pts(rotate_2d(pts)), label))

    # 6. Scale + noise
    augmented_rows.append((from_pts(add_noise(scale_variation(pts))), label))

    if (i+1) % 500 == 0:
        print(f"  Processed {i+1}/{len(X_raw)} samples...")

print(f"\nAugmented total: {len(augmented_rows)} samples")
print(f"Multiplier: {len(augmented_rows)/len(X_raw):.1f}x\n")

# ── Save ──────────────────────────────────────────────────────────────────
cols = [f"{c}{i}" for i in range(21) for c in ["x","y","z"]]
cols.append("label")

rows_arr = [(list(r), l) for r,l in augmented_rows]
X_aug    = np.array([r for r,_ in augmented_rows])
y_aug    = [l for _,l in augmented_rows]

aug_df   = pd.DataFrame(X_aug, columns=cols[:-1])
aug_df["label"] = y_aug

# Shuffle
aug_df = aug_df.sample(frac=1, random_state=42).reset_index(drop=True)

aug_df.to_csv(OUT_PATH, index=False)
print(f"[Saved] {OUT_PATH}")

# ── Per-class count ───────────────────────────────────────────────────────
print("\nPer-class sample count after augmentation:")
counts = aug_df["label"].value_counts().sort_index()
for label, count in counts.items():
    bar = "█" * (count // 20)
    print(f"  {label.upper()}: {count:4d}  {bar}")