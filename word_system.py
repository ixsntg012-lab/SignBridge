"""
ASL Communication System — Final
==================================
Two-way sign language communication tool.

MODE 1 — SIGN → TEXT  (deaf person signs)
    Hold sign 1 sec  → letter added to sentence
    SPACE            → add space
    BACKSPACE        → delete last char
    S                → speak sentence
    C                → clear
    1-5              → quick phrases
    TAB              → switch to Type Mode

MODE 2 — TYPE → SIGN  (hearing person types)
    Type any text    → ASL sign cards appear
    BACKSPACE        → delete
    S                → speak
    C                → clear
    TAB              → switch to Sign Mode

pip install opencv-python mediapipe scikit-learn joblib numpy pyttsx3
"""

import cv2
import numpy as np
import mediapipe as mp
import joblib
import time, sys, os
from collections import deque
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ── TTS ───────────────────────────────────────────────────────────────────
def speak(text):
    if not text.strip(): return
    try:
        import pyttsx3
        e = pyttsx3.init()
        e.setProperty('rate', 150)
        e.say(text)
        e.runAndWait()
    except Exception:
        try:
            if sys.platform == "win32":
                import win32com.client
                win32com.client.Dispatch("SAPI.SpVoice").Speak(text)
            elif sys.platform == "darwin":
                os.system(f'say "{text}"')
            else:
                os.system(f'espeak "{text}" 2>/dev/null')
        except Exception as e:
            print(f"[TTS Error] {e}")

# ── QUICK PHRASES (keys 1–5) ──────────────────────────────────────────────
QUICK_PHRASES = {
    '1': "Hello, how are you?",
    '2': "Thank you",
    '3': "I need help",
    '4': "Please wait",
    '5': "Nice to meet you",
}

# ── AUTOCOMPLETE WORD LIST ────────────────────────────────────────────────
WORDS = sorted({
    "hello","help","how","hi","happy","have","here","him","her","home",
    "the","thank","thanks","that","this","they","there","time","today",
    "sorry","see","sign","speak","school","stop","start","space",
    "please","people","place","pain",
    "want","where","when","what","who","with","water","we","wait",
    "are","and","again","all","also","ask",
    "need","nice","name","no","not","now",
    "my","me","meet","more","make","morning",
    "bye","back","bathroom","because","but","be","busy",
    "can","call","come","communicate","care",
    "do","day","dont","different",
    "eat","everything","everyone","excuse","emergency",
    "feel","fine","find","friend","from","food",
    "get","give","go","good","great","goodbye",
    "i","is","it","in","if",
    "know","kind","keep",
    "like","learn","listen","lost","language","late",
    "okay","ok","out","open","one",
    "right","read","repeat","restroom","ready",
    "take","talk","tell","together","tomorrow","translate","tired",
    "understand","us","use",
    "very","voice","visit",
    "yes","you","your","yesterday",
})

def autocomplete(partial, n=3):
    if not partial: return []
    return [w for w in WORDS if w.startswith(partial.lower())][:n]

# ── SIGN TILE ─────────────────────────────────────────────────────────────
def make_sign_tile(letter, size=100):
    """Show PNG from assets/signs/ if available, else draw letter card."""
    path = f"assets/signs/{letter.lower()}.png"
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            return cv2.resize(img, (size, size))
    # Fallback card
    tile = np.ones((size, size, 3), dtype=np.uint8) * 240
    cv2.rectangle(tile, (2,2), (size-2,size-2), (140,150,190), 2)
    f = cv2.FONT_HERSHEY_DUPLEX
    sc = min(2.8, size / 44.0)
    tw, th = cv2.getTextSize(letter.upper(), f, sc, 3)[0]
    cv2.putText(tile, letter.upper(),
                ((size-tw)//2, (size+th)//2),
                f, sc, (50,70,160), 3, cv2.LINE_AA)
    return tile

# ── PATHS ─────────────────────────────────────────────────────────────────
MODEL_PATH = "models/sign_model.pkl"
HAND_MODEL = "models/hand_landmarker.task"

if not os.path.exists(MODEL_PATH):
    sys.exit(f"[ERROR] {MODEL_PATH} not found. Run train_model.py first.")
if not os.path.exists(HAND_MODEL):
    sys.exit(f"[ERROR] {HAND_MODEL} not found.")

model    = joblib.load(MODEL_PATH)
base_opt = python.BaseOptions(model_asset_path=HAND_MODEL)
detector = vision.HandLandmarker.create_from_options(
    vision.HandLandmarkerOptions(base_options=base_opt, num_hands=1))

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]

# ── CONFIG ────────────────────────────────────────────────────────────────
CONF_THR  = 0.65
BUF_SIZE  = 10
HOLD_TIME = 1.0
COOLDOWN  = 0.4
WIN_W     = 1280
WIN_H     = 720
F  = cv2.FONT_HERSHEY_SIMPLEX
FB = cv2.FONT_HERSHEY_DUPLEX

# ── UI HELPERS ────────────────────────────────────────────────────────────
def txt(img, t, x, y, sc=0.55, col=(255,255,255), th=1, bold=False):
    cv2.putText(img, t, (x,y), FB if bold else F,
                sc, col, th, cv2.LINE_AA)

def panel(img, x, y, w, h, r=10, col=(22,24,40), a=0.88):
    ov = img.copy()
    cv2.rectangle(ov,(x+r,y),(x+w-r,y+h),col,-1)
    cv2.rectangle(ov,(x,y+r),(x+w,y+h-r),col,-1)
    for cx,cy in [(x+r,y+r),(x+w-r,y+r),(x+r,y+h-r),(x+w-r,y+h-r)]:
        cv2.circle(ov,(cx,cy),r,col,-1)
    cv2.addWeighted(ov,a,img,1-a,0,img)

def skeleton(frame, hand, w, h):
    for s,e in HAND_CONN:
        cv2.line(frame,
                 (int(hand[s].x*w),int(hand[s].y*h)),
                 (int(hand[e].x*w),int(hand[e].y*h)),
                 (180,100,220),2,cv2.LINE_AA)
    for lm in hand:
        cv2.circle(frame,(int(lm.x*w),int(lm.y*h)),4,(220,150,255),-1)

# ── SIGN MODE UI ──────────────────────────────────────────────────────────
def draw_sign_mode(frame, stable, conf, sentence, hold_prog, sugg):
    W, H = frame.shape[1], frame.shape[0]

    # ── Top panel ─────────────────────────────────────────────────────────
    panel(frame,12,6,W-24,122,10,(20,22,38),0.90)
    txt(frame,"ASL Recognition  —  SIGN → TEXT",
        28,40,sc=0.72,col=(100,180,255),th=2,bold=True)

    # Mode badge
    panel(frame,W-180,12,165,32,8,(30,80,45),0.92)
    txt(frame,"SIGN MODE",W-165,34,sc=0.50,col=(100,230,130),bold=True)

    # Sentence (truncate if long)
    disp = sentence if sentence else "—"
    if len(disp) > 52: disp = "..." + disp[-49:]
    txt(frame,f"Sentence:  {disp}",28,78,sc=0.63,col=(225,225,225))

    # ── Autocomplete suggestions ───────────────────────────────────────────
    if sugg:
        sx = 28
        txt(frame,"Suggestions:",sx,108,sc=0.38,col=(120,140,185))
        sx += 105
        for s in sugg:
            tw,_ = cv2.getTextSize(s,F,0.46,1)[0]
            panel(frame,sx,95,tw+18,24,5,(35,58,100),0.90)
            txt(frame,s,sx+9,112,sc=0.46,col=(160,200,255))
            sx += tw + 26

    # ── Right panel: current letter ────────────────────────────────────────
    px,py = W-205,135
    panel(frame,px,py,190,215,10,(22,24,40),0.90)

    if stable and stable != "?":
        txt(frame,stable.upper(),px+38,py+148,
            sc=4.5,col=(75,225,115),th=6,bold=True)
    else:
        txt(frame,"?",px+58,py+148,
            sc=4.5,col=(85,85,110),th=5,bold=True)

    # Confidence bar
    bx=px+12; by=py+168; bw=165; bh=10
    cv2.rectangle(frame,(bx,by),(bx+bw,by+bh),(40,42,60),-1)
    fill = int(bw * conf)
    bc = (75,225,115) if conf >= CONF_THR else (80,115,220)
    cv2.rectangle(frame,(bx,by),(bx+fill,by+bh),bc,-1)
    txt(frame,f"Confidence: {int(conf*100)}%",bx,by+28,sc=0.42,col=(165,185,220))

    # Hold progress arc
    if hold_prog > 0 and stable and stable != "?":
        c2x,c2y = px+95,py+88
        cv2.circle(frame,(c2x,c2y),45,(40,42,60),3)
        cv2.ellipse(frame,(c2x,c2y),(45,45),
                    -90,0,int(360*hold_prog),(75,225,115),3,cv2.LINE_AA)

    # ── Quick phrases panel ────────────────────────────────────────────────
    panel(frame,12,H-145,W-24,112,8,(20,22,38),0.88)
    txt(frame,"Quick Phrases:",22,H-122,sc=0.46,col=(130,150,200),bold=True)

    # Draw phrase buttons — 2 rows of 3 and 2
    row1 = list(QUICK_PHRASES.items())[:3]
    row2 = list(QUICK_PHRASES.items())[3:]
    for i,(k,phrase) in enumerate(row1):
        bx2 = 22 + i*((W-44)//3)
        short = phrase[:22]+"…" if len(phrase)>22 else phrase
        panel(frame,bx2,H-108,((W-50)//3)-4,28,6,(35,40,70),0.90)
        txt(frame,f"[{k}] {short}",bx2+8,H-88,sc=0.40,col=(190,210,255))

    for i,(k,phrase) in enumerate(row2):
        bx2 = 22 + i*((W-44)//2)
        short = phrase[:28]+"…" if len(phrase)>28 else phrase
        panel(frame,bx2,H-74,((W-50)//2)-4,28,6,(35,40,70),0.90)
        txt(frame,f"[{k}] {short}",bx2+8,H-54,sc=0.40,col=(190,210,255))

    # ── Hint bar ──────────────────────────────────────────────────────────
    panel(frame,12,H-38,W-24,32,6,(16,18,32),0.88)
    txt(frame,"SPACE=Space  BACK=Delete  S=Speak  C=Clear  TAB=Type Mode  1-5=Quick Phrase  ESC=Quit",
        22,H-18,sc=0.38,col=(130,148,190))

# ── TYPE MODE UI ──────────────────────────────────────────────────────────
def draw_type_mode(frame, typed):
    W, H = frame.shape[1], frame.shape[0]

    # Darken bg so sign tiles pop
    ov = frame.copy()
    cv2.rectangle(ov,(0,0),(W,H),(12,12,22),-1)
    cv2.addWeighted(ov,0.50,frame,0.50,0,frame)

    # ── Top panel ─────────────────────────────────────────────────────────
    panel(frame,12,6,W-24,82,10,(20,22,38),0.92)
    txt(frame,"TYPE → SIGN MODE  (Hearing person types here)",
        28,42,sc=0.68,col=(255,170,75),th=2,bold=True)
    panel(frame,W-180,12,165,32,8,(90,60,15),0.92)
    txt(frame,"TYPE MODE",W-165,34,sc=0.50,col=(255,190,90),bold=True)

    # ── Input box ─────────────────────────────────────────────────────────
    panel(frame,12,94,W-24,54,8,(28,30,50),0.92)
    cur = "|" if int(time.time()*2)%2==0 else " "
    # Truncate display if long
    disp = typed[-55:] if len(typed)>55 else typed
    txt(frame,f"> {disp}{cur}",26,130,sc=0.70,col=(235,235,235))

    # ── Sign tiles ────────────────────────────────────────────────────────
    letters = [c for c in typed.upper() if c.isalpha()]

    if letters:
        MAX_SHOW = 11
        show     = letters[-MAX_SHOW:]
        n        = len(show)
        tile_sz  = min(112, (W-60)//(n+1))
        tile_sz  = max(64, int(tile_sz))
        gap      = 6
        total    = n*tile_sz + (n-1)*gap
        sx       = (W-total)//2
        ty_top   = 160

        panel(frame,12,ty_top-14,W-24,tile_sz+65,10,(20,22,38),0.90)
        txt(frame,"ASL Hand Signs:",28,ty_top+10,sc=0.50,col=(145,165,220),bold=True)

        for i,letter in enumerate(show):
            tx = sx + i*(tile_sz+gap)
            tile = make_sign_tile(letter, tile_sz)

            # Place tile safely
            y1 = ty_top+24; y2 = y1+tile_sz
            x1 = tx;        x2 = tx+tile_sz
            y2 = min(y2, frame.shape[0])
            x2 = min(x2, frame.shape[1])
            if y1 < y2 and x1 < x2:
                frame[y1:y2, x1:x2] = tile[:y2-y1, :x2-x1]

            # Label under tile
            tw2,_ = cv2.getTextSize(letter,F,0.45,1)[0]
            txt(frame, letter, tx+(tile_sz-tw2)//2, y2+18,
                sc=0.45, col=(180,195,230))

        if len(letters) > MAX_SHOW:
            txt(frame,f"(showing last {MAX_SHOW} letters)",
                28, ty_top+tile_sz+52, sc=0.38, col=(100,110,150))

    else:
        panel(frame,12,155,W-24,130,10,(20,22,38),0.88)
        txt(frame,"Start typing — ASL hand signs appear here",
            W//2-230,228,sc=0.60,col=(90,100,145))
        txt(frame,"Deaf person can read the signs on this screen",
            W//2-240,264,sc=0.48,col=(80,92,130))

    # ── Hint bar ──────────────────────────────────────────────────────────
    panel(frame,12,H-38,W-24,32,6,(16,18,32),0.88)
    txt(frame,"Type  |  BACK=Delete  |  S=Speak  |  C=Clear  |  TAB=Sign Mode  |  ESC=Quit",
        22,H-18,sc=0.40,col=(130,148,190))

# ── CAMERA & STATE ────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  WIN_W)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WIN_H)

mode         = "SIGN"
sentence     = ""
typed_text   = ""
pred_buf     = deque(maxlen=BUF_SIZE)
stable_pred  = ""
conf         = 0.0
hold_start   = None
last_added   = ""
last_add_t   = 0.0
hold_prog    = 0.0
sugg         = []

print("="*55)
print("  ASL Communication System")
print("  TAB = switch modes  |  ESC = quit")
print("  Sign mode: hold letter 1 sec → added")
print("  Type mode: type text → see sign cards")
print("="*55)

# ── MAIN LOOP ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # ── SIGN MODE ─────────────────────────────────────────────────────────
    if mode == "SIGN":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = detector.detect(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))

        conf=0.0; stable_pred=""; hold_prog=0.0

        if res.hand_landmarks:
            hand = res.hand_landmarks[0]
            hp,wp,_ = frame.shape
            skeleton(frame, hand, wp, hp)

            wrist = hand[0]
            pts   = np.array([[lm.x-wrist.x, lm.y-wrist.y, lm.z-wrist.z]
                               for lm in hand], dtype=np.float32)
            md = np.max(np.linalg.norm(pts, axis=1))
            if md > 0: pts /= md
            X = pts.flatten().reshape(1,-1)

            probs = model.predict_proba(X)[0]
            conf  = float(probs.max())
            pred  = model.predict(X)[0]
            if conf < CONF_THR: pred = "?"
            pred_buf.append(pred)
            stable_pred = max(set(pred_buf), key=pred_buf.count)

            now = time.time()
            if stable_pred != "?" and \
                    pred_buf.count(stable_pred) >= BUF_SIZE*0.7:
                if hold_start is None: hold_start = now
                hold_prog = min((now-hold_start)/HOLD_TIME, 1.0)
                if hold_prog >= 1.0 and now-last_add_t > COOLDOWN:
                    if stable_pred != last_added or \
                            now-last_add_t > HOLD_TIME*2.5:
                        sentence   += stable_pred
                        last_added  = stable_pred
                        last_add_t  = now
                        hold_start  = None
                        pred_buf.clear()
                        print(f"[+] {stable_pred.upper()} → '{sentence}'")
            else:
                hold_start = None
        else:
            pred_buf.clear(); hold_start=None; last_added=""

        # Autocomplete on last partial word
        words = sentence.split(" ")
        sugg  = autocomplete(words[-1], 3) if words[-1] else []
        draw_sign_mode(frame, stable_pred, conf, sentence, hold_prog, sugg)

    # ── TYPE MODE ─────────────────────────────────────────────────────────
    else:
        draw_type_mode(frame, typed_text)

    cv2.imshow("ASL Communication System", frame)

    # ── KEY HANDLING ──────────────────────────────────────────────────────
    key = cv2.waitKey(1)
    if key == -1: continue

    # ESC → quit
    if key == 27:
        break

    # TAB → switch mode
    elif key == 9:
        mode = "TYPE" if mode=="SIGN" else "SIGN"
        pred_buf.clear(); hold_start=None
        print(f"[Mode] → {mode}")

    # S → speak
    elif key == ord('s'):
        t = sentence if mode=="SIGN" else typed_text
        if t.strip():
            print(f"[Speaking] {t}")
            speak(t)

    # C → clear
    elif key == ord('c'):
        if mode=="SIGN":
            sentence=""; last_added=""; pred_buf.clear(); sugg=[]
        else:
            typed_text=""

    # Sign mode only
    elif mode == "SIGN":
        if key == 32:                           # SPACE
            sentence += " "; last_added=""
        elif key == 8 and sentence:             # BACKSPACE
            sentence = sentence[:-1]
        elif chr(key) in QUICK_PHRASES:         # 1-5 quick phrases
            phrase    = QUICK_PHRASES[chr(key)]
            sentence  = phrase
            print(f"[Quick] {phrase}")
            speak(phrase)

    # Type mode only
    elif mode == "TYPE":
        if key == 8 and typed_text:             # BACKSPACE
            typed_text = typed_text[:-1]
        elif key == 32:                         # SPACE
            typed_text += " "
        elif 32 < key < 127:                    # printable
            typed_text += chr(key)

cap.release()
cv2.destroyAllWindows()