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
# These start as sane defaults based on real hand proportions, then
# AUTO-WIDEN during use to fit your specific hand/camera distance (see
# auto_ratio_min/auto_ratio_max in main()). They only ever expand, never
# shrink, so this can only get more accurate over time, never worse.
# (Old values of 0.15/0.50 were the bug: a relaxed, comfortable spread
# produces a ratio around 0.9-1.0 for most adult hands, so the volume
# clipped to 100% almost immediately and you only ever saw 0 or 100.)
RATIO_MIN_DEFAULT = 0.12    # fully pinched  -> 0%
RATIO_MAX_DEFAULT = 0.85    # fully spread   -> 100%
RATIO_ABS_MIN      = 0.04   # auto-min can never go below this (false-detection guard)
RATIO_ABS_MAX      = 1.60   # auto-max can never go above this (false-detection guard)
DEPTH_SMOOTH_FACTOR = 0.18   # how fast the hand-size (depth) reference is
                             # allowed to change — lower = ignores forward/
                             # back jitter more, but reacts slower to a
                             # genuine "I moved closer to the camera"
SMOOTH_FACTOR = 0.12    # exponential glide: lower = steadier, calmer
                         # volume movement. Raise toward 0.2-0.25 only
                         # if it starts feeling laggy/sluggish; this
                         # value favours a smooth, controlled rise/fall
                         # over an instant snap to the hand position
DEAD_ZONE     = 0.5     # % threshold before system volume moves
HISTORY_LEN   = 6        # rolling median window — longer = steadier,
                          # averages more recent frames before the value
                          # is even handed to the glide stage
MEDIAN_LEN    = 5        # inner median for spike removal

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
    # pycaw needs the COM apartment initialized on this thread before any
    # Activate()/EndpointVolume call - missing this is the #1 cause of
    # 'AudioDevice object has no attribute Activate' / EndpointVolume failures.
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except Exception as e:
        print(f"[WARNING] pythoncom.CoInitialize failed ({e}) - "
              f"if audio still fails below, run: pip install pywin32")

    from ctypes import cast, POINTER
    from comtypes import CLSCTX_ALL
    from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

    last_err = None

    # Path A (current pycaw): GetSpeakers() returns an AudioDevice wrapper
    # that exposes .Activate() directly.
    try:
        device = AudioUtilities.GetSpeakers()
        iface  = device.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(iface, POINTER(IAudioEndpointVolume))
        r = volume.GetVolumeRange()
        print(f"[INFO] Audio ready (Activate). dB range: {r[0]:.1f} to {r[1]:.1f}")
        return volume, r[0], r[1]
    except Exception as e:
        last_err = e

    # Path B (older pycaw): GetSpeakers() returns an object that already
    # exposes a high-level .EndpointVolume property.
    try:
        device = AudioUtilities.GetSpeakers()
        volume = device.EndpointVolume
        r = volume.GetVolumeRange()
        print(f"[INFO] Audio ready (EndpointVolume). dB range: {r[0]:.1f} to {r[1]:.1f}")
        return volume, r[0], r[1]
    except Exception as e:
        last_err = e

    # Path C: pull the raw COM device directly, in case
    # AudioUtilities.GetSpeakers() itself is returning the wrong wrapper type.
    try:
        devices = AudioUtilities.GetAllDevices()
        spk = next(d for d in devices if d.state == 1)  # DEVICE_STATE.ACTIVE
        iface  = spk._dev.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        volume = cast(iface, POINTER(IAudioEndpointVolume))
        r = volume.GetVolumeRange()
        print(f"[INFO] Audio ready (raw device). dB range: {r[0]:.1f} to {r[1]:.1f}")
        return volume, r[0], r[1]
    except Exception as e:
        last_err = e

    raise RuntimeError(
        f"All pycaw init paths failed: {last_err}\n"
        f"  Fix checklist:\n"
        f"  1. pip uninstall pycaw -y && pip install pycaw --upgrade\n"
        f"  2. pip install pywin32  (then run: python <venv>/Scripts/pywin32_postinstall.py -install, as admin)\n"
        f"  3. Run the terminal/VS Code as Administrator once to test"
    )

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

def get_hand_size(lms_px):
    """
    Robust scale reference for the hand, used to normalise the pinch
    distance against distance-from-camera (forward/back movement).

    A single bone (wrist -> middle MCP) is short and noisy: any small
    landmark jitter on either point swings the ratio a lot, and that
    noise gets *worse* the farther the hand is from the camera (fewer
    pixels -> more relative jitter). Averaging several independent
    bone spans across the palm cancels out per-landmark jitter and
    gives a much steadier "how big is this hand right now" number.
    """
    spans = (
        dist_px(lms_px[0], lms_px[5]),    # wrist -> index MCP
        dist_px(lms_px[0], lms_px[9]),    # wrist -> middle MCP
        dist_px(lms_px[0], lms_px[17]),   # wrist -> pinky MCP
        dist_px(lms_px[5], lms_px[17]),   # index MCP -> pinky MCP (palm width)
    )
    return max(sum(spans) / len(spans), 1.0)

def get_ratio(lms_px, hsize):
    pinch = dist_px(lms_px[4], lms_px[8])
    return pinch / hsize, lms_px[4], lms_px[8]

# ─────────────────────────────────────────────
# SMOOTHING PIPELINE
# ─────────────────────────────────────────────
class DepthSmoother:
    """
    Smooths the hand-size (depth) reference on its own, *before* it's
    used to divide the pinch distance into a ratio.

    This is the key fix for the "moving hand forward/back makes the
    volume drift" issue: previously the noisy raw hand-size was used
    immediately every frame, so any depth jitter went straight into
    the ratio and then had to be smoothed out *after the fact* (which
    just adds lag instead of fixing the cause). Filtering hand-size
    first means depth changes never reach the ratio calculation as
    noise in the first place — only as a genuine, slow scale change,
    which cancels out correctly.
    """
    def __init__(self):
        self.value = None

    def update(self, raw_size):
        if self.value is None:
            self.value = raw_size
        else:
            self.value += DEPTH_SMOOTH_FACTOR * (raw_size - self.value)
        return self.value


class VolumeSmoother:
    """
    3-stage pipeline:
      1. Median filter  — removes spike noise (outlier frames)
      2. Rolling mean   — averages out natural hand tremor
      3. Exponential    — creates the glide to system volume, tuned
                           to track the hand closely (fast sync) now
                           that the input ratio itself is already
                           clean thanks to DepthSmoother.
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

def draw_hud(frame, sv, fps, ratio, audio_ok, api, r_min, r_max):
    # HUD background
    roi  = frame[0:140, 0:230]
    dark = cv2.addWeighted(cv2.GaussianBlur(roi,(0,0),8),0.3,
                           np.full_like(roi,20),0.7, 0)
    frame[0:140, 0:230] = dark

    ac = (70,210,70) if audio_ok else (50,50,240)
    cv2.putText(frame, f"FPS   {fps:5.1f}",        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52, C_WHITE,       1)
    cv2.putText(frame, f"VOL   {int(round(sv))}%", (10, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.70, C_WHITE,       2)
    cv2.putText(frame, f"Ratio {ratio:.3f}  [{r_min:.2f}-{r_max:.2f}]",
                                                    (10, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (155,155,155), 1)
    cv2.putText(frame, "Audio ON" if audio_ok else "Audio OFF",
                                                    (10, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, ac,            1)
    cv2.putText(frame, f"API: {api}",               (10,118), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (90,90,90),    1)

    # Bottom bar
    roi2 = frame[CAM_HEIGHT-38:CAM_HEIGHT, 0:CAM_WIDTH]
    dark2 = cv2.addWeighted(cv2.GaussianBlur(roi2,(0,0),8),0.3,
                            np.full_like(roi2,20),0.7, 0)
    frame[CAM_HEIGHT-38:CAM_HEIGHT, 0:CAM_WIDTH] = dark2
    cv2.putText(frame, "Thumb-Index distance sets volume    Q = quit",
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

    # Camera — try the configured index first, then auto-fallback through
    # other common indices. CAM_INDEX=2 alone was a frequent hard-crash
    # point on machines where the webcam isn't at index 2.
    def try_open(idx):
        c = cv2.VideoCapture(idx, CAM_BACKEND)
        c.set(cv2.CAP_PROP_FRAME_WIDTH,  CAM_WIDTH)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        for _ in range(12):
            c.read()
        ok, t = c.read()
        if ok and t is not None and t.size > 0:
            return c
        c.release()
        return None

    print(f"[INFO] Opening camera index={CAM_INDEX}...")
    cap = try_open(CAM_INDEX)
    if cap is None:
        print(f"[WARNING] Camera index={CAM_INDEX} not responding, trying others...")
        for idx in [i for i in range(4) if i != CAM_INDEX]:
            print(f"[INFO] Trying camera index={idx}...")
            cap = try_open(idx)
            if cap is not None:
                print(f"[INFO] Camera found at index={idx}.")
                break
    if cap is None:
        print("[ERROR] No working camera found on indices 0-3. "
              "Check that a webcam is connected and not in use by another app.")
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

    smoother      = VolumeSmoother()
    depth_smoother = DepthSmoother()
    auto_ratio_min = RATIO_MIN_DEFAULT
    auto_ratio_max = RATIO_MAX_DEFAULT
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
                    hsize  = depth_smoother.update(get_hand_size(lms_px))
                    ratio, thumb, index = get_ratio(lms_px, hsize)
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
                    hsize  = depth_smoother.update(get_hand_size(lms_px))
                    ratio, thumb, index = get_ratio(lms_px, hsize)
                    draw_pinch(frame, thumb, index)

        # ── 3-stage smoothing ────────────────
        if hand_found:
            # Self-widening calibration: if you pinch tighter or spread wider
            # than what we've seen so far, stretch the bounds to fit it.
            # Sanity-clamped so a bad detection frame can't blow the range out.
            if RATIO_ABS_MIN < ratio < auto_ratio_min:
                auto_ratio_min = ratio
            if RATIO_ABS_MAX > ratio > auto_ratio_max:
                auto_ratio_max = ratio

            raw = float(np.clip(
                np.interp(ratio, [auto_ratio_min, auto_ratio_max], [0, 100]), 0, 100))
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
        draw_hud(frame, sv, fps, ratio, audio_ok, api_name, auto_ratio_min, auto_ratio_max)

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
