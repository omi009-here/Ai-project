# ============================================================
# HAND GESTURE VOLUME CONTROL — Maximum Stability Edition
# Camera: index=2, backend=CAP_ANY
# ============================================================
# pip install --upgrade opencv-python mediapipe pycaw numpy comtypes
# Press 'q' to quit
# ============================================================

import cv2
import numpy as np
import math
import time
import sys
from collections import deque

CAM_INDEX     = 2
CAM_BACKEND   = cv2.CAP_ANY
CAM_WIDTH     = 640
CAM_HEIGHT    = 480

# ── Stability pipeline ───────────────────────────────────────
RATIO_MIN     = 0.15    # fully pinched  -> 0%
RATIO_MAX     = 0.50    # fully spread   -> 100%
SMOOTH_FACTOR = 0.018   # ultra-gentle exponential glide
DEAD_ZONE     = 0.5     # % threshold before system volume moves
HISTORY_LEN   = 14      # rolling median window (larger = more stable)
MEDIAN_LEN    = 7       # inner median for spike removal

# ── Colours ─────────────────────────────────────────────────
C_YELLOW  = (0,   255, 255)
C_TEAL    = (0,   210, 210)
C_WHITE   = (255, 255, 255)
C_ORANGE  = (0,   165, 255)
C_BLUE    = (255,  80,   0)
C_DKGREY  = (28,   28,  28)
C_MIDGREY = (80,   80,  80)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (5,6),(6,7),(7,8),
    (9,10),(10,11),(11,12),
    (13,14),(14,15),(15,16),
    (17,18),(18,19),(19,20),
    (0,5),(5,9),(9,13),(13,17),(0,17)
]

# ─────────────────────────────────────────────
# AUDIO
# ─────────────────────────────────────────────
def init_audio():
    from pycaw.pycaw import AudioUtilities
    device = AudioUtilities.GetSpeakers()
    try:
        volume = device.EndpointVolume
        r = volume.GetVolumeRange()
        print(f"[INFO] Audio ready. dB range: {r[0]:.1f} to {r[1]:.1f}")
        return volume, r[0], r[1]
    except Exception:
        pass
    try:
        from ctypes import cast, POINTER
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import IAudioEndpointVolume
        iface  = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(iface, POINTER(IAudioEndpointVolume))
        r = volume.GetVolumeRange()
        print(f"[INFO] Audio ready (legacy). dB range: {r[0]:.1f} to {r[1]:.1f}")
        return volume, r[0], r[1]
    except Exception as e:
        raise RuntimeError(f"Both pycaw APIs failed: {e}")

# ─────────────────────────────────────────────
# MEDIAPIPE
# ─────────────────────────────────────────────
def init_tasks_api():
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    import urllib.request, os, tempfile
    model_path = os.path.join(tempfile.gettempdir(), "hand_landmarker.task")
    if not os.path.exists(model_path):
        print("[INFO] Downloading hand model (~9 MB, one-time)...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
            "hand_landmarker/float16/1/hand_landmarker.task", model_path)
        print("[INFO] Downloaded.")
    opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=model_path),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.75,
        min_hand_presence_confidence=0.75,
        min_tracking_confidence=0.65,
    )
    return mp_vision.HandLandmarker.create_from_options(opts), mp

def init_solutions_api():
    import mediapipe as mp
    mh    = mp.solutions.hands
    hands = mh.Hands(static_image_mode=False, max_num_hands=1,
                     min_detection_confidence=0.75, min_tracking_confidence=0.65)
    md    = mp.solutions.drawing_utils
    sl    = md.DrawingSpec(color=C_YELLOW, thickness=2, circle_radius=4)
    sc    = md.DrawingSpec(color=C_TEAL,   thickness=2)
    return hands, mh, md, sl, sc

# ─────────────────────────────────────────────
# GEOMETRY
# ─────────────────────────────────────────────
def dist_px(p1, p2):
    return math.hypot(p2[0]-p1[0], p2[1]-p1[1])

def get_ratio(lms_px):
    pinch = dist_px(lms_px[4], lms_px[8])
    hsize = max(dist_px(lms_px[0], lms_px[9]), 1.0)
    return pinch / hsize, lms_px[4], lms_px[8]

# ─────────────────────────────────────────────
# SMOOTHING PIPELINE
# ─────────────────────────────────────────────
class VolumeSmoother:
    """
    3-stage pipeline:
      1. Median filter  — removes spike noise (outlier frames)
      2. Rolling mean   — averages out natural hand tremor
      3. Exponential    — creates the buttery glide to system volume
    """
    def __init__(self):
        self.median_buf  = deque(maxlen=MEDIAN_LEN)
        self.history_buf = deque(maxlen=HISTORY_LEN)
        self.smooth      = 50.0
        self.last_set    = 50.0

    def update(self, raw_pct):
        # Stage 1: median spike removal
        self.median_buf.append(raw_pct)
        med = float(np.median(list(self.median_buf)))

        # Stage 2: rolling mean
        self.history_buf.append(med)
        avg = sum(self.history_buf) / len(self.history_buf)

        # Stage 3: exponential glide
        self.smooth += SMOOTH_FACTOR * (avg - self.smooth)
        return self.smooth

    def needs_update(self):
        return abs(self.smooth - self.last_set) > DEAD_ZONE

    def confirm_set(self):
        self.last_set = self.smooth

# ─────────────────────────────────────────────
# DRAWING
# ─────────────────────────────────────────────
def draw_landmarks_manual(frame, px):
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, px[a], px[b], C_TEAL, 2)
    for x, y in px:
        cv2.circle(frame, (x, y), 4, C_YELLOW, cv2.FILLED)

def draw_pinch(frame, p1, p2):
    mid = ((p1[0]+p2[0])//2, (p1[1]+p2[1])//2)
    cv2.line(frame,   p1,  p2,  C_ORANGE, 3)
    cv2.circle(frame, p1,  9,   C_BLUE,   cv2.FILLED)
    cv2.circle(frame, p2,  9,   C_BLUE,   cv2.FILLED)
    cv2.circle(frame, mid, 11,  C_ORANGE, cv2.FILLED)
    cv2.circle(frame, mid, 13,  C_WHITE,  2)

def vol_colour(pct):
    """Green at low, yellow at mid, orange-red at high."""
    if pct < 50:
        g = int(180 + pct * 1.4)
        return (0, min(g, 255), 50)
    else:
        t = (pct - 50) / 50.0
        r = int(t * 220)
        g = int(180 - t * 130)
        return (0, max(g, 0), min(r, 220))

def draw_volume_panel(frame, sv):
    """Pixel-smooth animated bar — driven by the float sv, not int."""
    PX, PY, PW, PH = 548, 55, 76, 410

    # Frosted panel
    roi     = frame[PY:PY+PH, PX:PX+PW]
    blurred = cv2.GaussianBlur(roi, (0,0), 8)
    dark    = cv2.addWeighted(blurred, 0.35, np.full_like(blurred, 20), 0.65, 0)
    frame[PY:PY+PH, PX:PX+PW] = dark

    # Bar geometry
    BX, BY = PX+18, PY+28
    BW, BH = 40,    PH-60

    # Track
    cv2.rectangle(frame, (BX,BY), (BX+BW, BY+BH), (55,55,55), -1)
    cv2.rectangle(frame, (BX,BY), (BX+BW, BY+BH), (85,85,85),  1)

    # Fill — pixel height from smooth float (no snapping)
    fill_h = max(int(round((sv / 100.0) * BH)), 0)
    fill_y = BY + BH - fill_h
    if fill_h > 0:
        colour = vol_colour(sv)
        cv2.rectangle(frame, (BX, fill_y), (BX+BW, BY+BH), colour, -1)
        # Bright top edge for polish
        cv2.line(frame, (BX, fill_y), (BX+BW, fill_y), C_WHITE, 1)

    # Tick marks + labels
    for pct in [0, 25, 50, 75, 100]:
        ty = BY + BH - int((pct/100)*BH)
        cv2.line(frame, (BX-6, ty), (BX, ty),      (110,110,110), 1)
        cv2.line(frame, (BX+BW, ty),(BX+BW+4, ty), (110,110,110), 1)
        cv2.putText(frame, str(pct), (BX-26, ty+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (130,130,130), 1)

    # Large % number
    label = f"{int(round(sv))}%"
    fs    = 0.85
    (tw,_),_ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
    cv2.putText(frame, label, (PX+(PW-tw)//2, PY+PH+26),
                cv2.FONT_HERSHEY_SIMPLEX, fs, vol_colour(sv), 2)

    # VOL header
    cv2.putText(frame, "VOL", (PX+24, PY+18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, C_WHITE, 1)

def draw_hud(frame, sv, fps, ratio, audio_ok, api):
    # HUD background
    roi  = frame[0:140, 0:230]
    dark = cv2.addWeighted(cv2.GaussianBlur(roi,(0,0),8),0.3,
                           np.full_like(roi,20),0.7, 0)
    frame[0:140, 0:230] = dark

    ac = (70,210,70) if audio_ok else (50,50,240)
    cv2.putText(frame, f"FPS   {fps:5.1f}",        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE,       1)
    cv2.putText(frame, f"VOL   {int(round(sv))}%", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_WHITE,       2)
    cv2.putText(frame, f"Ratio {ratio:.3f}",        (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (155,155,155), 1)
    cv2.putText(frame, "Audio ON" if audio_ok else "Audio OFF",
                                                    (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ac,            1)
    cv2.putText(frame, f"API: {api}",               (10,118), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90,90,90),    1)

    # Bottom bar
    roi2 = frame[CAM_HEIGHT-38:CAM_HEIGHT, 0:CAM_WIDTH]
    dark2 = cv2.addWeighted(cv2.GaussianBlur(roi2,(0,0),8),0.3,
                            np.full_like(roi2,20),0.7, 0)
    frame[CAM_HEIGHT-38:CAM_HEIGHT, 0:CAM_WIDTH] = dark2
    cv2.putText(frame, "PINCH = mute    SPREAD = max vol    Q = quit",
                (55, CAM_HEIGHT-12), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (170,170,170), 1)

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    # Audio
    audio_ok, vol_obj, db_min, db_max = False, None, -65.25, 0.0
    try:
        vol_obj, db_min, db_max = init_audio()
        audio_ok = True
    except Exception as e:
        print(f"[WARNING] Audio unavailable: {e}")

    # Camera
    print(f"[INFO] Opening camera index={CAM_INDEX}...")
    cap = cv2.VideoCapture(CAM_INDEX, CAM_BACKEND)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    for _ in range(12):
        cap.read()
    ret, test = cap.read()
    if not ret or test is None or test.size == 0:
        print("[ERROR] Camera not responding. Try changing CAM_INDEX.")
        sys.exit(1)
    print(f"[INFO] Camera ready: {int(cap.get(3))}x{int(cap.get(4))}")

    # MediaPipe
    use_tasks = False
    hands_obj = tasks_lander = mp_module = None
    mp_hands = mp_draw = sl = sc = None
    try:
        import mediapipe as mp
        _ = mp.solutions.hands
        hands_obj, mp_hands, mp_draw, sl, sc = init_solutions_api()
        api_name = "Solutions"
        print("[INFO] Using MediaPipe Solutions API.")
    except AttributeError:
        print("[INFO] Switching to Tasks API (Python 3.14 mode)...")
        try:
            tasks_lander, mp_module = init_tasks_api()
            use_tasks = True
            api_name  = "Tasks"
            print("[INFO] Tasks API ready.")
        except Exception as e:
            print(f"[ERROR] MediaPipe failed: {e}")
            cap.release()
            sys.exit(1)

    smoother  = VolumeSmoother()
    prev_time = time.time()
    ratio     = 0.0
    frame_ts  = 0

    print("[INFO] Ready — pinch to lower, spread to raise. Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None or frame.size == 0:
            blank = np.zeros((CAM_HEIGHT, CAM_WIDTH, 3), dtype=np.uint8)
            cv2.putText(blank, "Waiting for camera...", (140, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, C_TEAL, 2)
            cv2.imshow("Hand Volume Control  |  Q = quit", blank)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            continue

        frame = cv2.flip(frame, 1)

        # ── Detection ───────────────────────
        hand_found = False
        if not use_tasks:
            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands_obj.process(rgb)
            if results.multi_hand_landmarks:
                hand_found = True
                for hlms in results.multi_hand_landmarks:
                    mp_draw.draw_landmarks(frame, hlms, mp_hands.HAND_CONNECTIONS, sl, sc)
                    lms_px = [(int(lm.x*CAM_WIDTH), int(lm.y*CAM_HEIGHT)) for lm in hlms.landmark]
                    ratio, thumb, index = get_ratio(lms_px)
                    draw_pinch(frame, thumb, index)
        else:
            frame_ts += 33
            mp_img = mp_module.Image(image_format=mp_module.ImageFormat.SRGB,
                                     data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result = tasks_lander.detect_for_video(mp_img, frame_ts)
            if result.hand_landmarks:
                hand_found = True
                for hand in result.hand_landmarks:
                    lms_px = [(int(lm.x*CAM_WIDTH), int(lm.y*CAM_HEIGHT)) for lm in hand]
                    draw_landmarks_manual(frame, lms_px)
                    ratio, thumb, index = get_ratio(lms_px)
                    draw_pinch(frame, thumb, index)

        # ── 3-stage smoothing ────────────────
        if hand_found:
            raw = float(np.clip(np.interp(ratio, [RATIO_MIN, RATIO_MAX], [0, 100]), 0, 100))
            sv  = smoother.update(raw)
        else:
            sv  = smoother.smooth   # freeze while no hand

        # ── Set system volume ────────────────
        if audio_ok and vol_obj and smoother.needs_update():
            try:
                db = float(np.interp(smoother.smooth, [0, 100], [db_min, db_max]))
                vol_obj.SetMasterVolumeLevel(db, None)
                smoother.confirm_set()
            except Exception as ex:
                print(f"[WARNING] Volume set failed: {ex}")
                audio_ok = False

        # ── Render ──────────────────────────
        now       = time.time()
        fps       = 1.0 / max(now - prev_time, 1e-6)
        prev_time = now

        draw_volume_panel(frame, sv)
        draw_hud(frame, sv, fps, ratio, audio_ok, api_name)

        cv2.imshow("Hand Volume Control  |  Q = quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if hands_obj:    hands_obj.close()
    if tasks_lander: tasks_lander.close()
    print("[INFO] Exited cleanly.")

if __name__ == "__main__":
    main()
