# SignBridge — ASL Fingerspelling Communication System

A real-time two-way communication tool that bridges the gap between the **Deaf community and hearing individuals** using American Sign Language (ASL) fingerspelling.

Built with Python, MediaPipe, and Machine Learning — runs entirely on a standard webcam, no special hardware required.

![Demo](screenshot.png)

---

## Who Is This For?

This system is designed primarily as a **bridge tool for hearing individuals** who do not know sign language but need to communicate with Deaf or Hard-of-Hearing people.

| User | Mode | What They Do |
|---|---|---|
| Hearing person (doesn't know ASL) | **Type → Sign** | Types text → ASL hand sign cards appear on screen → Deaf person reads the signs |
| Hearing person learning ASL | **Sign → Text** | Practices signs → system recognizes and shows the letter → verifies correctness |
| Educator / Parent | **Both modes** | Uses as a learning and communication aid |

> **Honest note:** Deaf individuals who are fluent in ASL do not need this system to understand their own signs — they already know them. The value of this tool is for the hearing side of the conversation, and for learners building their fingerspelling skills.

---

## Features

| Feature | Description |
|---|---|
| Two-way communication | Sign→Text mode and Type→Sign mode in one app |
| Real-time recognition | 30fps hand landmark detection via MediaPipe |
| A–Y alphabet support | 24 static ASL letters (J and Z are motion-based) |
| Sign card display | Type any text → see the corresponding ASL hand sign for each letter |
| Autocomplete | Suggests words as you sign — speeds up sentence building |
| Quick phrases | Press 1–5 for common phrases (Hello, Thank you, I need help…) |
| Smoothed predictions | 10-frame majority-vote buffer eliminates flickering |
| Duplicate suppression | Hold-timer logic prevents unintended repeated letters |
| Text-to-speech | Speaks the sentence aloud — cross-platform |
| Data augmentation | 10x dataset expansion via noise, flip, scale, rotation |
| Model evaluation | Per-class accuracy report + confusion matrix |

---

## How It Works

```
Webcam
  │
  ▼
MediaPipe Hand Landmarker
(21 hand keypoints × 3D coordinates)
  │
  ▼
Feature Engineering
(Wrist-relative normalization + scale normalization)
  │
  ▼
Soft Voting Ensemble (Random Forest × 2)
  │
  ▼
10-frame Majority Vote Buffer
  │
  ▼
Hold Timer + Duplicate Suppression
  │
  ▼
Sentence Builder → Text-to-Speech
```

---

## Two Modes

### Mode 1 — SIGN → TEXT
Deaf person or ASL learner signs into the webcam.
System recognizes the letter, builds words, speaks the sentence.

```
Hold any ASL sign for 1 second → letter added to sentence
```

### Mode 2 — TYPE → SIGN
Hearing person types normally.
The corresponding ASL hand sign card appears for each letter.
Deaf person reads the signs on screen.

```
Type "hello" → H, E, L, L, O sign cards appear side by side
```

Switch between modes with **TAB**.

---

## Dataset

- **Collected by:** Manually using `collect_data.py`
- **Size:** ~150–200 samples per letter × 24 letters ≈ 3,600+ samples
- **After augmentation:** ~36,000+ samples (10x expansion)
- **Features:** 21 landmarks × 3 coordinates (x, y, z) = 63 features per frame
- **Normalization:** Wrist-relative translation + max-distance scale normalization

### Augmentation techniques applied:
| Technique | Purpose |
|---|---|
| Gaussian noise (×3) | Simulates natural hand tremor |
| Horizontal flip | Adds left-hand variants |
| Scale variation (×2) | Handles different distances from camera |
| 2D rotation (×2) | Handles wrist tilt variation |
| Combined (noise + scale) | Edge case coverage |

---

## Model

**Soft Voting Ensemble — Random Forest × 2**
- Two Random Forest classifiers with different hyperparameters
- Probabilities averaged → smoother, more calibrated confidence scores
- Evaluated with 5-fold stratified cross-validation

See `models/eval_report.txt` and `models/confusion_matrix.png` after running `train_model.py`.

---

## Tech Stack

| Component | Technology |
|---|---|
| Hand Tracking | MediaPipe Hand Landmarker |
| ML Model | Scikit-learn (Random Forest Ensemble) |
| Feature Engineering | Wrist-relative 3D normalization |
| Data Augmentation | NumPy geometric transforms |
| Computer Vision | OpenCV |
| Speech Output | pyttsx3 (cross-platform) |

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/Sign-language-recognition.git
cd Sign-language-recognition
pip install -r requirements.txt
```

Download `hand_landmarker.task` from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it in the `models/` folder.

> **Note:** `sign_model.pkl` is not included in the repo (large file).
> Run `python train_model.py` to generate it after collecting data.

---

## Usage

```bash
# Step 1 — Collect data (only if retraining)
python collect_data.py

# Step 2 — Clean dataset
python fix_dataset.py

# Step 3 — Augment dataset
python augment_data.py

# Step 4 — Train model
python train_model.py

# Step 5 — Run the system
python word_system.py
```

---

## Controls

| Key | Action |
|---|---|
| `TAB` | Switch between Sign mode and Type mode |
| `SPACE` | Add space between words |
| `BACKSPACE` | Delete last character |
| `S` | Speak current sentence aloud |
| `C` | Clear sentence |
| `1` | Quick phrase: "Hello, how are you?" |
| `2` | Quick phrase: "Thank you" |
| `3` | Quick phrase: "I need help" |
| `4` | Quick phrase: "Please wait" |
| `5` | Quick phrase: "Nice to meet you" |
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
├── augment_data.py              ← data augmentation pipeline
├── train_model.py               ← model training + evaluation
├── word_system.py               ← main application
│
├── requirements.txt
├── screenshot.png
└── README.md
```

---

## Limitations & Future Work

- **J and Z** require motion/trajectory tracking — not currently supported.
  Future plan: LSTM-based sequence model for dynamic signs.
- Recognition accuracy may vary with lighting conditions and hand size diversity in training data.
- Expanding dataset to 500+ samples per letter would improve generalization across users.
- Potential extension: full word-level sign language recognition using sequence models.

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science

---

## License

MIT License