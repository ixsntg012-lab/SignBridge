# SignBridge 🤟

<div align="center">

**Real-time, Multi-Agent ASL Communication System**

*Bridging the gap between Deaf and hearing individuals — with LLM-powered natural language understanding*

![Python](https://img.shields.io/badge/Python-3.9+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Hand_Tracking-0097A7?style=for-the-badge)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Groq](https://img.shields.io/badge/Groq_AI-LLM-F55036?style=for-the-badge)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-5C3EE8?style=for-the-badge&logo=opencv)
![Accuracy](https://img.shields.io/badge/Accuracy-98.91%25-brightgreen?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

</div>

---

## What is SignBridge?

SignBridge is a real-time, two-way ASL fingerspelling communication tool designed to help **hearing individuals communicate with Deaf or Hard-of-Hearing people** — without knowing sign language.

It combines **computer vision, classical machine learning, and an LLM-powered language agent** in a multi-agent pipeline: hand signs are detected, converted to raw text, then an AI agent infers the intended sentence — correcting missed letters and grammar — before speaking it aloud.

It runs entirely on a standard webcam. No special hardware. No internet required for sign detection (only for the language refinement step).

> 🎥 **Demo:** See [`demo.gif`](#) below for a recorded walkthrough. This is a desktop, webcam-based application, so it does not have a hosted live demo — cloud servers don't have physical webcam access. See [Installation](#installation) to run it locally.

---

## Multi-Agent Architecture

SignBridge is built as three coordinating agents rather than a single monolithic script:

```
┌─────────────────┐     ┌──────────────────┐     ┌───────────────────┐
│   Agent 1        │     │   Agent 2         │     │   Agent 3          │
│   Vision          │ ──▶ │   Language (LLM)  │ ──▶ │   Response          │
│                    │     │                   │     │                    │
│  MediaPipe +       │     │  Groq LLM infers  │     │  Text-to-Speech +   │
│  Random Forest     │     │  intended sentence│     │  on-screen display  │
│  ensemble detects  │     │  from garbled/    │     │  of BOTH raw and    │
│  letters from      │     │  incomplete raw   │     │  refined text       │
│  hand signs         │     │  text              │     │                    │
└─────────────────┘     └──────────────────┘     └───────────────────┘
```

---

## Two Modes

### Mode 1 — SIGN → TEXT
Deaf person or ASL learner signs into the webcam. The system:
1. Recognizes each letter (Vision Agent)
2. Builds a raw sentence
3. On pressing `S`, the **Language Agent** infers the intended sentence — even from incomplete or garbled input (e.g. a misdetected letter or a learner's imperfect signing) — and speaks it aloud
4. Both the raw and refined sentence are shown on screen, so the Deaf person (who can't hear the spoken output) can also see that the system understood them correctly

```
Sign "i wan o hme" (letters missed/misdetected)
  → Raw:      i wan o hme
  → Refined:  I want to go home.   (spoken aloud + shown on screen)
```

### Mode 2 — TYPE → SIGN
Hearing person types normally. The corresponding ASL hand sign card appears on screen for the Deaf person to read.

```
Type "hello" → H E L L O sign cards appear side by side on screen
```

Switch between modes with **TAB**.

---

## Results

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **98.91%** |
| 5-Fold CV Accuracy | 98.79% ± 0.18% |
| Dataset (raw) | 3,677 samples across 24 letters |
| Dataset (augmented) | 40,447 samples (11x expansion) |
| Model | Soft Voting Ensemble — Random Forest × 2 |
| Input features | 63 (21 landmarks × x, y, z) |
| Frame rate | 30fps real-time |
| Letters supported | 24 (A–Y; J and Z require motion — see Future Work) |

---

## How It Works

```
Webcam Frame (30fps)
        │
        ▼
MediaPipe Hand Landmarker          ┐
(21 hand keypoints — 3D coords)    │
        │                          │  AGENT 1 — Vision
        ▼                          │
Feature Engineering                │
(Wrist-relative + scale norm)      │
        │                          │
        ▼                          │
Soft Voting Ensemble (RF × 2)      │
        │                          │
        ▼                          │
10-frame Majority Vote Buffer      ┘
        │
        ▼
Hold Timer + Duplicate Suppression
        │
        ▼
Raw Sentence Builder + Autocomplete
        │
        ▼
Groq LLM — Language Agent          ┐  AGENT 2 — Language
(infers intended sentence from     │
 garbled/incomplete raw text)      ┘
        │
        ▼
Text-to-Speech + On-screen display ┐  AGENT 3 — Response
(both raw AND refined text shown)  ┘
```

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Hand Tracking | MediaPipe Hand Landmarker |
| ML Model | Scikit-learn — Soft Voting Random Forest Ensemble |
| Language Agent | Groq API (LLM) — intent inference from garbled input |
| Feature Engineering | Wrist-relative 3D normalization + scale normalization |
| Data Augmentation | NumPy geometric transforms (6 techniques) |
| Computer Vision | OpenCV |
| Speech Output | pyttsx3 (cross-platform TTS) |

---

## Dataset & Augmentation

**Self-collected** using a custom webcam collection tool (`collect_data.py`), with targeted additional collection for underperforming letters (see the 'T' letter case study below).

| Augmentation Technique | Purpose |
|------------------------|---------|
| Gaussian noise (×3) | Simulates natural hand tremor |
| Horizontal flip | Adds left-hand variants |
| Scale variation (×2) | Handles different distances from camera |
| 2D rotation (×2) | Handles wrist tilt variation |
| Noise + scale combined | Edge case coverage |

**Result:** 3,677 raw samples → 40,447 augmented samples (11x expansion)

---

## Case Study: Diagnosing and Fixing the 'T' Letter

During real-world testing, the letter **'T'** was inconsistently detected compared to other letters.

**Root cause:** insufficient and low-variation training samples for 'T' relative to other letters.

**Fix applied:**
1. Collected 50+ additional 'T' samples across varied hand angles and lighting conditions using `collect_data.py`
2. Re-ran the full pipeline: clean → augment → retrain
3. Result: 'T' precision improved to **99.8%**, recall to **100%** (F1-score: 0.999)

**Lesson learned:** per-class sample quality and diversity matters as much as overall dataset size — a model can have high aggregate accuracy while still failing badly on individual classes with weaker data.

---

## Installation

```bash
git clone https://github.com/ixsntg012-lab/SignBridge.git
cd SignBridge
pip install -r requirements.txt
```

Download `hand_landmarker.task` from [MediaPipe Models](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) and place it in the `models/` folder.

Create a `.env` file for the Language Agent:
```
GROQ_API_KEY=your_groq_api_key
```
Get a free Groq API key: [console.groq.com](https://console.groq.com)

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
| `S` | Refine sentence via Language Agent + speak it aloud (Sign mode) / Speak as-is (Type mode) |
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
├── language_agent.py            ← Agent 2: LLM-based sentence refinement
├── word_system.py               ← main application (orchestrates all agents)
│
├── .env                         ← GROQ_API_KEY (not committed — see .gitignore)
├── requirements.txt
└── README.md
```

---

## Limitations & Future Work

**Motion-based signs (current limitation)**

The current system recognizes **static hand poses** (a single frame is enough to classify a letter). This works well for 24 of the 26 ASL letters, but two categories of real ASL are motion-based and are not yet supported:

- **Letters J and Z** — these involve tracing a shape in the air rather than holding a static pose
- **Common word-level signs** (e.g. "Thank you", "Please", "Sorry") — fluent ASL signers use one fluid, whole-word sign for these rather than fingerspelling every letter, which is both faster and more natural than what this system currently supports

Both are the same underlying technical gap: **static single-frame classification cannot capture motion over time.**

- [ ] LSTM/sequence-based model trained on landmark trajectories over multiple frames, covering both J/Z and a starter set of common word-level signs (Thank you, Please, Hello, Sorry, Help)

**Other future work**

- [ ] Accuracy may vary with extreme lighting conditions or hand size diversity — expanding to 500+ samples per letter across more diverse users would improve generalization
- [ ] Browser-based version using `streamlit-webrtc` or TensorFlow.js, so it can run as a hosted, interactive demo without a local install
- [ ] Confidence calibration for the Language Agent's inference step (flagging low-confidence corrections for human review)

---

## Author

**Swetha Kiran Veernapu**
MS Computer Science

---

## License

MIT License
