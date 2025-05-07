import csv
import json
from pathlib import Path

import cv2
import numpy as np

FPS = 120

BALL_MARKUP_PATH = Path("ball_markup.json")
EVENT_MARKUP_PATH = Path("events_markup.json")

with BALL_MARKUP_PATH.open() as f:
    BALL_LOOKUP = json.load(f)

with EVENT_MARKUP_PATH.open() as f:
    EVENT_LOOKUP = json.load(f)


def detect_ball(frame_idx):
    key = str(frame_idx)
    entry = BALL_LOOKUP.get(key)
    if entry is None:
        return None
    return entry["x"], entry["y"]


def transform_ball_position(x, y, transform_matrix):
    src = np.float32([[[x, y]]])
    dst = cv2.perspectiveTransform(src, transform_matrix)
    tx, ty = dst[0][0]
    return tx, ty


def calculate_velocity(positions, frames, fps=FPS):
    if len(positions) < 2 or len(frames) < 2:
        return 0, 0
    dx = positions[-1][0] - positions[0][0]
    dy = positions[-1][1] - positions[0][1]
    dt = (frames[-1] - frames[0]) / fps
    if dt == 0:
        return 0, 0
    vx = dx / dt
    vy = dy / dt
    return vx, vy


def find_line_interval(a, b, c_values, x0, y0):
    if len(c_values) != 3:
        print("Error: Exactly three c values must be provided.")
        raise Exception
    epsilon = 1e-9
    vals = [a * x0 + b * y0 + c for c in c_values]
    value_with_index = [(vals[i], i) for i in range(3)]

    for val, index in value_with_index:
        if abs(val) < epsilon:
            if index == 0:
                return "low"
            elif index == 1:
                return "mid"
            else:
                return "high"

    sorted_values_with_indices = sorted(value_with_index)
    val1_s, idx1_s = sorted_values_with_indices[0]
    val2_s, idx2_s = sorted_values_with_indices[1]
    val3_s, idx3_s = sorted_values_with_indices[2]

    if (val1_s < -epsilon and val2_s > epsilon) or (
        val1_s > epsilon and val2_s < -epsilon
    ):
        if max(idx1_s, idx2_s) == 1:
            return "mid"
        elif max(idx1_s, idx2_s) == 2 and min(idx1_s, idx2_s) == 0:
            return "high"
        elif max(idx1_s, idx2_s) == 2 and min(idx1_s, idx2_s) == 1:
            return "high"
    if (val2_s < -epsilon and val3_s > epsilon) or (
        val2_s > epsilon and val3_s < -epsilon
    ):
        if max(idx2_s, idx3_s) == 1:
            return "mid"
        elif max(idx2_s, idx3_s) == 2 and min(idx2_s, idx3_s) == 0:
            return "high"
        elif max(idx2_s, idx3_s) == 2 and min(idx2_s, idx3_s) == 1:
            return "high"

    if vals[0] < -epsilon:
        return "low"
    elif vals[0] > epsilon:
        return "high"

    print("Undefined Zone")
    return None


def find_line_interval_from_points(lines_points, x0, y0):
    if len(lines_points) != 3:
        return "Error: Exactly three lines must be provided."
    point1_line1 = lines_points[0][0]
    point2_line1 = lines_points[0][1]
    x1_l1, y1_l1 = point1_line1[0], point1_line1[1]
    x2_l1, y2_l1 = point2_line1[0], point2_line1[1]
    dx = x2_l1 - x1_l1
    dy = y2_l1 - y1_l1
    a = dy
    b = -dx
    c_values = [-a * lp[0][0] - b * lp[0][1] for lp in lines_points]
    return find_line_interval(a, b, c_values, x0, y0)


def get_ball_height_category(tx, ty, break_line1_pts, break_line2_pts, break_line3_pts):
    lines = [
        np.array(break_line1_pts, dtype=np.float32),
        np.array(break_line2_pts, dtype=np.float32),
        np.array(break_line3_pts, dtype=np.float32),
    ]
    return find_line_interval_from_points(lines, float(tx), float(ty))


def get_event(frame_idx):
    key = str(frame_idx)
    entry = EVENT_LOOKUP.get(key)
    if entry is None or entry == "empty_event":
        return None
    return entry


x1_orig, y1_orig = 528, 611
x2_orig, y2_orig = 1269, 617
x3_orig, y3_orig = 1418, 782
x4_orig, y4_orig = 395, 788

xt1_warp, yt1_warp = 528, 611
xt2_warp, yt2_warp = 1269, 611
xt3_warp, yt3_warp = 1418, 782
xt4_warp, yt4_warp = 395, 782

avg_inset = ((xt1_warp - xt4_warp) + (xt3_warp - xt2_warp)) / 2.0
xt1_adj = xt4_warp + avg_inset
xt2_adj = xt3_warp - avg_inset

src_points = np.float32(
    [[x1_orig, y1_orig], [x2_orig, y2_orig], [x3_orig, y3_orig], [x4_orig, y4_orig]]
)
dst_points = np.float32(
    [
        [xt1_adj, yt1_warp],
        [xt2_adj, yt2_warp],
        [xt3_warp, yt3_warp],
        [xt4_warp, yt4_warp],
    ]
)
front_transform, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 0.5)

edge1_vis_unreshaped = np.array(
    [[xt1_adj, yt1_warp], [xt4_warp, yt4_warp], [xt4_warp, 0], [xt1_adj, 0]], np.int32
)
edge1_vis = edge1_vis_unreshaped.reshape((-1, 1, 2))
edge2_vis_unreshaped = np.array(
    [[xt2_adj, yt2_warp], [xt3_warp, yt3_warp], [xt3_warp, 0], [xt2_adj, 0]], np.int32
)
edge2_vis = edge2_vis_unreshaped.reshape((-1, 1, 2))

cap = cv2.VideoCapture("./game_4.mp4")

step1 = 80
step2 = 160
# step3 = 180
e1_break1_pts = np.array([[xt1_adj, yt1_warp], [xt4_warp, yt4_warp]], np.int32)
e1_break2_pts = np.array(
    [[xt1_adj, yt1_warp - step1], [xt4_warp, yt4_warp - step1]], np.int32
)
e1_break3_pts = np.array(
    [[xt1_adj, yt1_warp - step2], [xt4_warp, yt4_warp - step2]], np.int32
)
# e1_break3_pts = np.array([[xt1_adj, yt1_warp - step3], [xt4_warp, yt4_warp - step3]], np.int32)

e2_break1_pts = np.array([[xt2_adj, yt2_warp], [xt3_warp, yt3_warp]], np.int32)
e2_break2_pts = np.array(
    [[xt2_adj, yt2_warp - step1], [xt3_warp, yt3_warp - step1]], np.int32
)
e2_break3_pts = np.array(
    [[xt2_adj, yt2_warp - step2], [xt3_warp, yt3_warp - step2]], np.int32
)
# e2_break3_pts = np.array([[xt2_adj, yt2_warp - step3], [xt3_warp, yt3_warp - step3]], np.int32)

e1_break1_vis = e1_break1_pts.reshape((-1, 1, 2))
e1_break2_vis = e1_break2_pts.reshape((-1, 1, 2))
e1_break3_vis = e1_break3_pts.reshape((-1, 1, 2))
e2_break1_vis = e2_break1_pts.reshape((-1, 1, 2))
e2_break2_vis = e2_break2_pts.reshape((-1, 1, 2))
e2_break3_vis = e2_break3_pts.reshape((-1, 1, 2))

# Define X-ranges for selecting break lines based on ball's tx
# Range for Edge 2 side (xt2_adj, xt3_warp) - if ball here, use Edge 1's break lines
x_range_edge2_side = (min(xt2_adj, xt3_warp), max(xt2_adj, xt3_warp))
# Range for Edge 1 side (xt4_warp, xt1_adj) - if ball here, use Edge 2's break lines
x_range_edge1_side = (min(xt4_warp, xt1_adj), max(xt4_warp, xt1_adj))

dataset_file = Path("ball_bounce_dataset.csv")
with dataset_file.open("w", newline="") as csvfile:
    fieldnames = [
        "bounce_frame",
        "vx_before",
        "vy_before",
        # "vx_after",
        # "vy_after",
        "bounce_x",
        "bounce_y",
        "height_category",
    ]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()

    ball_positions = []
    ball_frames = []
    bounce_frames = []
    frame_idx = 0

    data = None
    prev_tx = None
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        event = get_event(frame_idx)
        if event == "bounce":
            bounce_frames.append(frame_idx)

        ball_coords = detect_ball(frame_idx)
        # img = cv2.warpPerspective(frame, front_transform, (1920, 1080))

        # cv2.polylines(img, [edge1_vis], True, (0, 255, 255))
        # cv2.polylines(img, [edge2_vis], True, (0, 255, 255))
        # cv2.polylines(img, [e1_break1_vis], True, (0, 255, 255))
        # cv2.polylines(img, [e2_break1_vis], True, (0, 255, 255))
        # cv2.polylines(img, [e1_break2_vis], True, (0, 255, 255))
        # cv2.polylines(img, [e2_break2_vis], True, (0, 255, 255))
        # cv2.polylines(img, [e1_break3_vis], True, (0, 255, 255))
        # cv2.polylines(img, [e2_break3_vis], True, (0, 255, 255))

        height_cat_display = "N/A"
        current_break_lines = None

        if ball_coords is not None:
            x, y = ball_coords
            tx, ty = transform_ball_position(x, y, front_transform)
            ball_positions.append((tx, ty))
            ball_frames.append(frame_idx)
            # cv2.circle(img, (int(tx), int(ty)), 10, (0, 0, 255), -1)

            # Conditionally select break lines
            window = 100
            if x_range_edge2_side[0] - window <= tx <= x_range_edge2_side[1] + window:
                current_break_lines = (e1_break1_pts, e1_break2_pts, e1_break3_pts)
                # cv2.putText(img, "Using E1 Breaks", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,0), 1) # Debug
            elif x_range_edge1_side[0] - window <= tx <= x_range_edge1_side[1] + window:
                current_break_lines = (e2_break1_pts, e2_break2_pts, e2_break3_pts)
                # cv2.putText(img, "Using E2 Breaks", (150, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,200), 1) # Debug

            if current_break_lines:
                height_cat_display = get_ball_height_category(
                    tx, ty, *current_break_lines
                )
            if data and height_cat_display:
                data["height_category"] = height_cat_display
                writer.writerow(data)
                print(data)
                data = None
            # cv2.putText(
            #     img,
            #     f"Height: {height_cat_display}",
            #     (50, 50),
            #     cv2.FONT_HERSHEY_SIMPLEX,
            #     1,
            #     (255, 255, 255),
            #     2,
            # )

        if frame_idx > 0 and frame_idx in bounce_frames:
            frame_window = 60
            pre_bounce_indices = [
                i
                for i, f in enumerate(ball_frames)
                if frame_idx - frame_window <= f < frame_idx
            ]
            # post_bounce_indices = [
            #     i for i, f in enumerate(ball_frames) if frame_idx < f <= frame_idx + frame_window
            # ]

            # print(pre_bounce_indices, post_bounce_indices)
            if len(pre_bounce_indices) >= 1:  # and len(post_bounce_indices) >= 1:
                pre_pos = [ball_positions[i] for i in pre_bounce_indices]
                pre_frames = [ball_frames[i] for i in pre_bounce_indices]
                # post_pos = [ball_positions[i] for i in post_bounce_indices]
                # post_frames = [ball_frames[i] for i in post_bounce_indices]
                vx_before, vy_before = calculate_velocity(pre_pos, pre_frames)
                # vx_after, vy_after = calculate_velocity(post_pos, post_frames)

                height_category_bounce = "N/A"
                bounce_ball_idx = next(
                    (i for i, f in enumerate(ball_frames) if f == frame_idx), None
                )
                # print(bounce_ball_idx)
                if bounce_ball_idx is not None:
                    bounce_tx, bounce_ty = ball_positions[bounce_ball_idx]
                    bounce_break_lines = None
                    if x_range_edge2_side[0] <= bounce_tx <= x_range_edge2_side[1]:
                        bounce_break_lines = (
                            e1_break1_pts,
                            e1_break2_pts,
                            e1_break3_pts,
                        )
                    elif x_range_edge1_side[0] <= bounce_tx <= x_range_edge1_side[1]:
                        bounce_break_lines = (
                            e2_break1_pts,
                            e2_break2_pts,
                            e2_break3_pts,
                        )

                    if bounce_break_lines:
                        height_category_bounce = get_ball_height_category(
                            bounce_tx, bounce_ty, *bounce_break_lines
                        )

                    # print(
                    #     {
                    #         "bounce_frame": frame_idx,
                    #         "vx_before": vx_before,
                    #         "vy_before": vy_before,
                    #         # "vx_after": vx_after,
                    #         # "vy_after": vy_after,
                    #         "bounce_x": bounce_tx,
                    #         "bounce_y": bounce_ty,
                    #         "height_category": height_category_bounce,
                    #     }
                    # )
                    data = {
                        "bounce_frame": frame_idx,
                        "vx_before": vx_before,
                        "vy_before": vy_before,
                        # "vx_after": vx_after,
                        # "vy_after": vy_after,
                        "bounce_x": bounce_tx,
                        "bounce_y": bounce_ty,
                    }
                    # writer.writerow(
                    #     {
                    #         "bounce_frame": frame_idx,
                    #         "vx_before": vx_before,
                    #         "vy_before": vy_before,
                    #         # "vx_after": vx_after,
                    #         # "vy_after": vy_after,
                    #         "bounce_x": bounce_tx,
                    #         "bounce_y": bounce_ty,
                    #         "height_category": height_category_bounce,
                    #     }
                    # )
                    # cv2.putText(
                    #     img,
                    #     f"Pre-b vel: ({vx_before:.1f},{vy_before:.1f})",
                    #     (50, 100),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.8,
                    #     (255, 255, 255),
                    #     2,
                    # )
                    # cv2.putText(
                    #     img,
                    #     f"Post-b vel: ({vx_after:.1f},{vy_after:.1f})",
                    #     (50, 150),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.8,
                    #     (255, 255, 255),
                    #     2,
                    # )
                    # cv2.putText(
                    #     img,
                    #     f"Bounce H: {height_category_bounce}",
                    #     (50, 200),
                    #     cv2.FONT_HERSHEY_SIMPLEX,
                    #     0.8,
                    #     (255, 255, 255),
                    #     2,
                    # )

        frame_idx += 1
        # cv2.imshow("Coach view with ball tracking", img)
        # if cv2.waitKey(1) & 0xFF == 27:
        #     break

    cap.release()
    cv2.destroyAllWindows()
print(f"Dataset created and saved to {dataset_file}")
