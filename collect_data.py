"""
Data Collection — Sign Language Dataset
=========================================
Collect hand landmark samples for each ASL letter.
Saves normalized (x,y,z) coordinates of 21 landmarks to CSV.

Controls:
  A-Z  → collect sample for that letter (J and Z skipped)
  Q    → quit
"""

import cv2
import csv
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_DIR   = "data"
CSV_PATH   = os.path.join(DATA_DIR, "signs.csv")
MODEL_PATH = "models/hand_landmarker.task"

os.makedirs(DATA_DIR, exist_ok=True)

LETTERS = [l for l in "abcdefghijklmnopqrstuvwxyz" if l not in ["j","z"]]

# ── MediaPipe ────────────────────────────────────────────────────────────
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options      = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector     = vision.HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ── CSV header ────────────────────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        header = [f"{c}{i}" for i in range(21) for c in ["x","y","z"]]
        header.append("label")
        writer.writerow(header)

# ── Load existing counts ──────────────────────────────────────────────────
counts = {l: 0 for l in LETTERS}
if os.path.exists(CSV_PATH):
    with open(CSV_PATH) as f:
        for row in csv.DictReader(f):
            if row.get("label") in counts:
                counts[row["label"]] += 1

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

FONT = cv2.FONT_HERSHEY_SIMPLEX
print("Press A-Z to collect samples | Q to quit")

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)

    hand_detected = bool(result.hand_landmarks)

    if hand_detected:
        hand = result.hand_landmarks[0]
        h, w, _ = frame.shape
        for s,e in HAND_CONNECTIONS:
            x1,y1 = int(hand[s].x*w), int(hand[s].y*h)
            x2,y2 = int(hand[e].x*w), int(hand[e].y*h)
            cv2.line(frame,(x1,y1),(x2,y2),(180,100,220),2,cv2.LINE_AA)
        for lm in hand:
            cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),4,(220,150,255),-1)

    # ── UI ───────────────────────────────────────────────────────────────
    overlay = frame.copy()
    cv2.rectangle(overlay,(0,0),(frame.shape[1],95),(20,20,35),-1)
    cv2.addWeighted(overlay,0.82,frame,0.18,0,frame)

    cv2.putText(frame,"ASL Data Collection",(20,32),FONT,0.8,(100,180,255),2,cv2.LINE_AA)

    status = "Hand Detected ✓" if hand_detected else "No Hand"
    sc = (80,220,120) if hand_detected else (80,80,200)
    cv2.putText(frame,status,(20,62),FONT,0.5,sc,1,cv2.LINE_AA)

    # Sample counts — two rows
    row1 = LETTERS[:12]; row2 = LETTERS[12:]
    for row_idx, row_letters in enumerate([row1, row2]):
        line = "  ".join(f"{l.upper()}:{counts[l]}" for l in row_letters)
        cv2.putText(frame, line, (20, 115 + row_idx*25),
                    FONT, 0.45, (85,140,85), 1, cv2.LINE_AA)

    cv2.putText(frame,"Press A-Z to save sample | Q to quit",
                (20, frame.shape[0]-15), FONT, 0.45, (150,165,200), 1, cv2.LINE_AA)

    cv2.imshow("Data Collection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

    if hand_detected and ord('a') <= key <= ord('z'):
        letter = chr(key)
        if letter in LETTERS:
            hand = result.hand_landmarks[0]
            row  = [c for lm in hand for c in [lm.x, lm.y, lm.z]]
            row.append(letter)
            with open(CSV_PATH, "a", newline="") as f:
                csv.writer(f).writerow(row)
            counts[letter] += 1
            print(f"[Saved] {letter.upper()}  (total: {counts[letter]})")

cap.release()
cv2.destroyAllWindows()
print("\nFinal counts:")
for l,c in counts.items():
    print(f"  {l.upper()}: {c}")