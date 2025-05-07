import json
from pathlib import Path

import cv2
import numpy as np

# table corners
# u0, v0 = 528, 611
# u1, v1 = 1269, 611
# u2, v2 = 1418, 782
# u3, v3 = 395, 788
#
# ball_radius_px_ref = 7
#
# h, w = 1920, 1080
# f_px = 0.9 * w  # assume ~60° diagonal FOV
# K = np.array([[f_px, 0, w / 2], [0, f_px, h / 2], [0, 0, 1]], dtype=np.float32)
#
# obj = np.float32([[0, 0, 0], [2740, 0, 0], [2740, 1525, 0], [0, 1525, 0]])
# img = np.float32([[u0, v0], [u1, v1], [u2, v2], [u3, v3]])
#
# _, rvec, tvec = cv2.solvePnP(obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE)
# R, _ = cv2.Rodrigues(rvec)  # 3×3
# C = -R.T @ tvec  # camera centre in world coords


# -----------  LOAD THE MARK‑UP ONCE  -----------------
BALL_MARKUP_PATH = Path("ball_markup.json")
EVENT_MARKUP_PATH = Path("events_markup.json")

with BALL_MARKUP_PATH.open() as f:
    BALL_LOOKUP = json.load(f)  # keys are strings: "0", "1", ...

with EVENT_MARKUP_PATH.open() as f:
    EVENT_LOOKUP = json.load(f)


def detect_ball(frame_idx):
    """
    Parameters
    ----------
    frame_idx : int
        Zero‑based index of the current frame in the video stream.

    Returns
    -------
    (cx, cy, r_px)  or  None
        - centre x, centre y (pixels)
        - dummy radius (pixels), needed only to keep call‑signature compatible
        If the frame isn’t annotated, returns None.
    """
    key = str(frame_idx)
    entry = BALL_LOOKUP.get(key)
    if entry is None:
        return None  # → skip height update this frame
    return entry["x"], entry["y"], 16  # radius placeholder


def get_event(frame_idx):
    key = str(frame_idx)
    entry = EVENT_LOOKUP.get(key)
    if entry is None:
        for i in range(10):
            entry = EVENT_LOOKUP.get(frame_idx + i)
            if entry:
                break
            entry = EVENT_LOOKUP.get(frame_idx - i)
            if entry:
                break
    if entry is None:
        return None
    return entry


# def pixel_to_height(u, v, R, t, K, ball_radius_px=10):
#     """Return the ball height z (mm) given its pixel centre."""
#     # 1) pixel → unit ray in camera coords
#     uv1 = np.array([u, v, 1.0])
#     ray_cam = np.linalg.inv(K) @ uv1
#     ray_cam /= np.linalg.norm(ray_cam)
#
#     # 2) convert ray to world coords
#     ray_world = R.T @ ray_cam
#     cam_world = (-R.T @ t).reshape(-1)
#
#     # 3) find intersection with table plane z=0 → ground point
#     s_ground = -cam_world[2] / ray_world[2]
#     ground_point = cam_world + s_ground * ray_world   # (xg, yg, 0)
#
#     # 4) find s_ball so pixel projects to ball radius above ground
#     # In a pure pinhole, s scales inversely with image radius:
#     s_ball = s_ground * (ball_radius_px_ref / ball_radius_px)
#     ball_point = cam_world + s_ball * ray_world       # (xb, yb, zb)
#
#     return ball_point[2]          # zb is the height
#


# ---------- ONE‑TIME CALIBRATION (fill these four tuples) ----------

## test 1
# xN, yN_table = 808, 679  # 1️⃣ table edge near
# yN_net = 620             # 2️⃣ net top   near    (same xN)
#
# xF, yF_table = 886, 559  # 3️⃣ table edge far
# yF_net = 520             # 4️⃣ net top   far     (same xF)
#
# NET_H_MM = 152.5         # ITTF net height

## game 4
xN, yN_table = 918, 755  # 1️⃣ table edge near
yN_net = 805  # 2️⃣ net top   near    (same xN)

xF, yF_table = 905, 560  # 3️⃣ table edge far
yF_net = 610  # 4️⃣ net top   far     (same xF)

NET_H_MM = 152.5  # ITTF net height

# ----- Pre‑compute linear models -----
# (a) mm/px as a function of horizontal position x
mm_px_near = NET_H_MM / (yN_table - yN_net)
mm_px_far = NET_H_MM / (yF_table - yF_net)


def mm_per_px(x):
    """Linear interpolation between the near and far scale factors."""
    return mm_px_near + (mm_px_far - mm_px_near) * (x - xN) / (xF - xN)


# (b) table‑edge (z = 0) image row as a function of x
table_slope = (yF_table - yN_table) / (xF - xN)


def y_table(x):
    """Row index of the table‑top edge at horizontal position x."""
    return yN_table + table_slope * (x - xN)


# ---------- RUNTIME LOOP ----------
frame_idx = 0
alpha = 0.25  # smoother
smoothed_h = 0.0

cap = cv2.VideoCapture("./game_4.mp4")  # or path to your video

while True:
    ok, frame = cap.read()
    if not ok:
        break

    # ---- 1. detect the ball centre ------------------------------
    detection = detect_ball(frame_idx)  # <-- your favourite detector
    event = get_event(frame_idx)
    frame_idx += 1

    colour = (255, 255, 255)

    if event and event != "empty_event":
        cv2.putText(
            frame,
            f"{event}",
            (1000, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            2.0,
            colour,
            4,
        )

    cv2.putText(
        frame,
        f"{frame_idx}",
        (1400, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        colour,
        4,
    )

    if detection is None:
        cv2.imshow("coach view", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
        continue

    cx, cy, r_px = detection

    # ---- 2. perspective‑aware height ----------------------------
    px_scale = mm_per_px(cx)  # mm per pixel at this x
    table_y = y_table(cx)  # table row at this x

    height_mm = (table_y - cy) * px_scale  # raw height

    # ---- 3. smooth ----------------------------------------------
    smoothed_h = alpha * height_mm + (1 - alpha) * smoothed_h

    # ---- 4. bucket for coaching ---------------------------------
    if smoothed_h < 30:
        bucket, colour = "skid", (0, 255, 0)
    elif smoothed_h < 150:
        bucket, colour = "low", (0, 200, 200)
    elif smoothed_h < 300:
        bucket, colour = "medium", (0, 128, 255)
    else:
        bucket, colour = "high", (0, 0, 255)

    cv2.putText(
        frame,
        f"{bucket}  {smoothed_h/100:.2f}cm",
        (40, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.0,
        colour,
        4,
    )

    cv2.imshow("coach view", frame)
    if cv2.waitKey(2000 // 120) & 0xFF == 27:  # Esc to quit
        break

cap.release()
cv2.destroyAllWindows()
