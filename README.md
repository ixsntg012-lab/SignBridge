# SignBridge 🤟

<div align="center">

**Real-time two-way ASL Fingerspelling Communication System**

*Bridging the gap between Deaf and hearing individuals — no special hardware required*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-0097A7?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-98.96%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## What is SignBridge?

SignBridge is a real-time, two-way ASL fingerspelling communication tool designed to help **hearing individuals communicate with Deaf or Hard-of-Hearing people** — without knowing sign language.

It runs entirely on a standard webcam. No special hardware. No internet required.

---

## Two Modes

### Mode 1 — SIGN → TEXT
Deaf person or ASL learner signs into the webcam.
System recognizes each letter, builds words, and speaks the sentence aloud.

```
Hold any ASL sign for 1 second → letter confirmed → sentence built → spoken aloud
```

### Mode 2 — TYPE → SIGN
Hearing person types normally.
The corresponding ASL hand sign card appears on screen for the Deaf person to read.

```
Type "hello" → H E L L O sign cards appear side by side on screen
```

Switch between modes with **TAB**.

---

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.96%** |
| Validation | 5-fold Stratified Cross-Validation |
| Dataset (raw) | ~3,600 samples (150–200 per letter × 24 letters) |
| Dataset (augmented) | ~36,000 samples (10x expansion) |
| Model | Soft Voting Ensemble — Random Forest × 2 |
| Input features | 63 (21 landmarks × x, y, z) |
| Frame rate | 30fps real-time |
| Letters supported | 24 (A–Y; J and Z require motion — future work) |

---

## How It Works

```
Webcam Frame (30fps)
        │
        ▼
MediaPipe Hand Landmarker
(21 hand keypoints — 3D coordinates)
        │
        ▼
Feature Engineering
(Wrist-relative + scale normalization → 63 features)
        │
        ▼
Soft Voting Ensemble (Random Forest × 2)
        │
        ▼
10-frame Majority Vote Buffer
(Eliminates flickering from natural hand tremor)
        │
        ▼
Hold Timer + Duplicate Suppression
        │
        ▼
Sentence Builder → Autocomplete → Text-to-Speech
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand Tracking | MediaPipe Hand Landmarker |
| ML Model | Scikit-learn — Soft Voting Random Forest Ensemble |
| Feature Engineering | Wrist-relative 3D normalization + scale normalization |
| Data Augmentation | NumPy geometric transforms (6 techniques) |
| Computer Vision | OpenCV |
| Speech Output | pyttsx3 (cross-platform TTS) |

---

## Dataset & Augmentation

**Self-collected** using a custom webcam collection tool (`collect_data.py`).

| Augmentation Technique | Purpose |
|------------------------|---------|
| Gaussian noise (×3) | Simulates natural hand tremor |
| Horizontal flip | Adds left-hand variants |
| Scale variation (×2) | Handles different distances from camera |
| 2D rotation (×2) | Handles wrist tilt variation |
| Noise + scale combined | Edge case coverage |

**Result:** 3,600 raw samples → 36,000 augmented samples (10x expansion)

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/SignBridge.git
cd SignBridge
pip install -r requirements.txt
```

Download `hand_landmarker.task` from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it in the `models/` folder.

> **Note:** `sign_model.pkl` is not included (large file). Run `python train_model.py` to generate it after collecting data.

---

## Usage

```bash
# Step 1 — Collect data
python collect_data.py

# Step 2 — Clean dataset
python fix_dataset.py

# Step 3 — Augment dataset
python argument_data.py

# Step 4 — Train model
python train_model.py

# Step 5 — Run the system
python word_system.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `TAB` | Switch between Sign mode and Type mode |
| `SPACE` | Add space between words |
| `BACKSPACE` | Delete last character |
| `S` | Speak current sentence aloud |
| `C` | Clear sentence |
| `1–5` | Quick phrases (Hello / Thank you / I need help / Please wait / Nice to meet you) |
| `ESC` | Quit |

---

## Project Structure

```
SignBridge/
│
├── data/
│   ├── signs.csv                ← raw collected samples
│   └── signs_augmented.csv      ← augmented dataset (generated)
│
├── models/
│   ├── hand_landmarker.task     ← MediaPipe model (download separately)
│   ├── sign_model.pkl           ← trained classifier (generated)
│   ├── eval_report.txt          ← accuracy report (generated)
│   └── confusion_matrix.png     ← confusion matrix (generated)
│
├── collect_data.py              ← webcam data collection tool
├── fix_dataset.py               ← dataset cleaning
├── argument_data.py             ← data augmentation pipeline
├── train_model.py               ← model training + evaluation
├── word_system.py               ← main application
│
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

- **J and Z** require motion/trajectory tracking — not currently supported.
  Future plan: LSTM-based sequence model for dynamic signs.
- Accuracy may vary with extreme lighting conditions or hand size diversity.
- Expanding to 500+ samples per letter and more diverse users would improve generalization.
- Web version using TensorFlow.js — run entirely in browser without installation.

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science

---

## License

MIT License
