import csv
import json
from pathlib import Path
import numpy as np
import cv2  # Import cv2 at the top level
import argparse  # Import argparse for command line argument parsing

# Constants
FPS = 120  # Same FPS as in the original script

def transform_ball_position(x, y, transform_matrix):
    """Apply perspective transform to ball position"""
    src = np.float32([[[x, y]]])
    dst = cv2.perspectiveTransform(src, transform_matrix)
    tx, ty = dst[0][0]
    return tx, ty

def calculate_velocity(positions, frames, fps=FPS):
    """Calculate velocity from positions and frames"""
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
    """Find which interval a point is in relative to three lines"""
    if len(c_values) != 3:
        print("Error: Exactly three c values must be provided.")
        raise Exception
    epsilon = 1e-9
    vals = [a * x0 + b * y0 + c for c in c_values]
    value_with_index = [(vals[i], i) for i in range(3)]

    # on line
    if abs(vals[2]) < epsilon: # On the highest break-line (line 2)
        return "low"
    if abs(vals[1]) < epsilon: # On the middle break-line (line 1)
        return "mid"
    if abs(vals[0]) < epsilon: # On the lowest break-line (line 0)
        return "high"

    if vals[2] < -epsilon:  # Point is physically above the highest line (line 2).
        return "low"
    # If not above line 2, it's at or below line 2.
    # Now check against line 1.
    elif vals[1] < -epsilon: # Point is physically above the middle line (line 1)
                             # (and implicitly at or below line 2).
                             # This is the zone between line 1 and line 2.
        return "mid"
    # If not above line 1, it's at or below line 1.
    # Now check against line 0.
    elif vals[0] < -epsilon: # Point is physically above the lowest line (line 0)
                             # (and implicitly at or below line 1).
                             # This is the zone between line 0 and line 1.
        return "mid"
    # If not above line 0, it's at or below line 0.
    else: # Point is physically below the lowest line (line 0)
          # (i.e., vals[0] > epsilon, since the on-line case for vals[0] is handled).
        return "high"
    # sorted_values_with_indices = sorted(value_with_index)
    # val1_s, idx1_s = sorted_values_with_indices[0]
    # val2_s, idx2_s = sorted_values_with_indices[1]
    # val3_s, idx3_s = sorted_values_with_indices[2]

    # # Modified logic to achieve more balanced height categorization
    # if (val1_s < -epsilon and val2_s > epsilon) or (val1_s > epsilon and val2_s < -epsilon):
    #     if max(idx1_s, idx2_s) == 1:
    #         return "mid"  # Between first and second lines
    #     elif max(idx1_s, idx2_s) == 2 and min(idx1_s, idx2_s) == 0:
    #         return "high"  # Above highest line
    #     elif max(idx1_s, idx2_s) == 2 and min(idx1_s, idx2_s) == 1:
    #         return "mid"  # Changed from "high" to "mid" to balance categories

    # if (val2_s < -epsilon and val3_s > epsilon) or (val2_s > epsilon and val3_s < -epsilon):
    #     if max(idx2_s, idx3_s) == 1:
    #         return "mid"
    #     elif max(idx2_s, idx3_s) == 2 and min(idx2_s, idx3_s) == 0:
    #         return "high"
    #     elif max(idx2_s, idx3_s) == 2 and min(idx2_s, idx3_s) == 1:
    #         return "mid"

    # if vals[0] < -epsilon:
    #     return "low"
    # elif vals[0] > epsilon:
    #     if vals[1] < 0:  # If below second line
    #         print("alert")
    #         return "low"  # More conservative "low" classification
    #     else:
    #         return "high"

    # if vals[2] > epsilon:  # Above the highest line (High-Break)
    #     return "high"
    # elif vals[1] > epsilon:  # Above Mid-Break line (but not High-Break) -> between Mid-Break and High-Break
    #     return "mid"
    # elif vals[0] > epsilon:  # Above Net (but not Mid-Break) -> between Net and Mid-Break
    #     return "mid"
    # else:  # Below Net (or on the net if not caught by earlier 'on line' checks)
    #     return "low"


    # print("Undefined Zone")
    # return None

def find_line_interval_from_points(lines_points, x0, y0):
    """Find which interval a point is in based on line points"""
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
    """Determine ball height category"""
    lines = [
        np.array(break_line1_pts, dtype=np.float32),
        np.array(break_line2_pts, dtype=np.float32),
        np.array(break_line3_pts, dtype=np.float32),
    ]
    return find_line_interval_from_points(lines, float(tx), float(ty))

def calculate_post_bounce_height(bounce_frame, frames, BALL_LOOKUP, front_transform,
                                x_min, x_max, break_lines):
    """
    Track the ball after bounce and determine its height
    until it reaches table edge or changes velocity direction

    Args:
        bounce_frame: Frame number where bounce occurs
        frames: Sorted list of all valid frame numbers
        BALL_LOOKUP: Dictionary of ball positions
        front_transform: Transformation matrix for perspective transform
        x_min, x_max: Table edge x coordinates
        break_lines: Triple of break line points for height categorization

    Returns:
        The height category based on post-bounce trajectory
    """
    # Get frames after bounce - don't limit to a fixed number
    post_frames = [f for f in frames if bounce_frame < f < bounce_frame + 120]
    post_frames.sort()  # Ensure frames are in order
    if not post_frames:
        return None  # No frames to analyze

    # Track positions and calculate frame-by-frame velocities
    positions = []
    heights = []
    last_velocity_y = None
    velocity_direction_changed = False

    for frame in post_frames:
        frame_str = str(frame)
        if frame_str in BALL_LOOKUP and BALL_LOOKUP[frame_str]["x"] != -1:
            x = BALL_LOOKUP[frame_str]["x"]
            y = BALL_LOOKUP[frame_str]["y"]
            tx, ty = transform_ball_position(x, y, front_transform)
            positions.append((tx, ty))
            heights.append(ty)  # Lower y value means higher position

            # Calculate vertical velocity direction
            if len(positions) >= 2:
                current_velocity_y = ty - positions[-2][1]  # Positive means moving down

                # Check if velocity direction changed
                if last_velocity_y is not None:
                    # If sign changed from positive to negative or vice versa
                    if (last_velocity_y * current_velocity_y < 0) and abs(current_velocity_y) > 1.0:
                        velocity_direction_changed = True
                        break

                last_velocity_y = current_velocity_y

            # Check if ball reached table edge
            if tx <= x_min or tx >= x_max:
                break

            # Check if we have enough points to make a determination
            if len(positions) >= 3 and (velocity_direction_changed or len(positions) >= 10):
                # If we've collected enough data points, we can make a determination
                break

    if len(positions) < 2:  # Need at least 2 points for analysis
        return None

    # Find the highest point (minimum y value)
    min_height = float('inf')  # Track minimum height (maximum y position)
    min_height_index = 0

    for i in range(len(positions)):
        if heights[i] < min_height:
            min_height = heights[i]
            min_height_index = i

    # Use the minimum height point as our key position for categorization
    best_position = positions[min_height_index]

    # Determine height category using the best position
    height_category = get_ball_height_category(best_position[0], best_position[1], *break_lines)

    return height_category

def main():
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Process ball bounce data.")
    parser.add_argument(
        "--corners",
        type=str,
        default="corners.json",
        help="Path to the corners coordinates JSON file"
    )
    parser.add_argument(
        "--ball_markup",
        type=str,
        default="ball_markup.json",
        help="Path to the ball markup JSON file"
    )
    parser.add_argument(
        "--event_markup",
        type=str,
        default="events_markup.json",
        help="Path to the events markup JSON file"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ball_bounce_dataset_from_json.csv",
        help="Output CSV file for the dataset"
    )
    args = parser.parse_args()

    # Load JSON files
    BALL_MARKUP_PATH = Path(args.ball_markup)
    EVENT_MARKUP_PATH = Path(args.event_markup)
    CORNERS_PATH = Path(args.corners)

    with BALL_MARKUP_PATH.open() as f:
        BALL_LOOKUP = json.load(f)

    with EVENT_MARKUP_PATH.open() as f:
        EVENT_LOOKUP = json.load(f)

    with CORNERS_PATH.open() as f:
        CORNERS = json.load(f)

    # Define transformation matrix for perspective transform - same as in original script
    x1_orig, y1_orig = CORNERS["x1_orig"], CORNERS["y1_orig"]
    x2_orig, y2_orig = CORNERS["x2_orig"], CORNERS["y2_orig"]
    x3_orig, y3_orig = CORNERS["x3_orig"], CORNERS["y3_orig"]
    x4_orig, y4_orig = CORNERS["x4_orig"], CORNERS["y4_orig"]

    # xt1_warp, yt1_warp = 500, 536
    # xt2_warp, yt2_warp = 1287, 536
    # xt3_warp, yt3_warp = 1418, 418
    # xt4_warp, yt4_warp = 395, 418

    avg_inset = ((x1_orig - x4_orig) + (x3_orig - x2_orig)) / 2.0
    xt1_adj = x4_orig + avg_inset
    xt2_adj = x3_orig - avg_inset

    xt3_warp = x3_orig
    xt4_warp = x4_orig

    yt1_warp = yt2_warp = (y1_orig + y2_orig) // 2
    yt3_warp = yt4_warp = (y3_orig + y4_orig) // 2

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
    front_transform, _ = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 0.5)

    # Define break lines with adjusted step sizes for better height categorization
    s = yt3_warp - yt2_warp
    step1 = s * 0.4
    step2 = s * 1.2

    e1_break1_pts = np.array([[xt1_adj, yt1_warp], [xt4_warp, yt4_warp]], np.int32)
    e1_break2_pts = np.array(
        [[xt1_adj, yt1_warp - step1], [xt4_warp, yt4_warp - step1]], np.int32
    )
    e1_break3_pts = np.array(
        [[xt1_adj, yt1_warp - step2], [xt4_warp, yt4_warp - step2]], np.int32
    )

    e2_break1_pts = np.array([[xt2_adj, yt2_warp], [xt3_warp, yt3_warp]], np.int32)
    e2_break2_pts = np.array(
        [[xt2_adj, yt2_warp - step1], [xt3_warp, yt3_warp - step1]], np.int32
    )
    e2_break3_pts = np.array(
        [[xt2_adj, yt2_warp - step2], [xt3_warp, yt3_warp - step2]], np.int32
    )

    # Define X-ranges for selecting break lines based on ball's tx
    x_range_edge2_side = (min(xt2_adj, xt3_warp), max(xt2_adj, xt3_warp))
    x_range_edge1_side = (min(xt4_warp, xt1_adj), max(xt4_warp, xt1_adj))

    # Define the middle of the court to determine which break lines to use for balls in the middle
    mid_court_x = (max(x_range_edge1_side[1], x_range_edge2_side[1]) + min(x_range_edge1_side[0], x_range_edge2_side[0])) / 2

    # Define the table boundaries
    table_min_x = min(xt4_warp, xt1_adj, xt2_adj, xt3_warp)
    table_max_x = max(xt4_warp, xt1_adj, xt2_adj, xt3_warp)

    # Convert string frame indices to integers for sorting
    frames = [int(frame) for frame in BALL_LOOKUP.keys() if BALL_LOOKUP[frame]["x"] != -1]
    frames.sort()  # Ensure frames are in order

    # Find bounce frames
    bounce_frames = [int(frame) for frame in EVENT_LOOKUP.keys() if EVENT_LOOKUP[frame] == "bounce"]
    bounce_frames.sort()

    # Track height category distribution
    height_counts = {"low": 0, "mid": 0, "high": 0, "N/A": 0}

    # Prepare dataset
    dataset_file = Path(args.output)
    with dataset_file.open("w", newline="") as csvfile:
        fieldnames = [
            "bounce_frame",
            "vx_before",
            "vy_before",
            "bounce_x",
            "bounce_y",
            "height_category",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # Process each bounce frame
        for bounce_frame in bounce_frames:
            bounce_frame_str = str(bounce_frame)

            # Get ball position at bounce
            if bounce_frame_str not in BALL_LOOKUP or BALL_LOOKUP[bounce_frame_str]["x"] == -1:
                print(f"No valid ball position for bounce frame {bounce_frame}")
                continue

            bounce_x = BALL_LOOKUP[bounce_frame_str]["x"]
            bounce_y = BALL_LOOKUP[bounce_frame_str]["y"]

            # Transform ball position
            bounce_tx, bounce_ty = transform_ball_position(bounce_x, bounce_y, front_transform)

            # Get frames before bounce for velocity calculation
            frame_window = 60  # Same as in original script
            pre_bounce_frames = [f for f in frames if bounce_frame - frame_window <= f < bounce_frame]

            if len(pre_bounce_frames) < 2:
                print(f"Not enough frames before bounce at frame {bounce_frame} for velocity calculation")
                continue

            # Collect positions for velocity calculation
            pre_positions = []
            for frame in pre_bounce_frames:
                frame_str = str(frame)
                x = BALL_LOOKUP[frame_str]["x"]
                y = BALL_LOOKUP[frame_str]["y"]
                tx, ty = transform_ball_position(x, y, front_transform)
                pre_positions.append((tx, ty))

            # Calculate velocity
            vx_before, vy_before = calculate_velocity(pre_positions, pre_bounce_frames)

            # Determine which break lines to use based on ball position
            bounce_break_lines = None

            # Use a more refined approach to determine which break lines to use
            if bounce_tx <= mid_court_x:
                # Ball is on the left side of the court, use Edge 2's break lines
                bounce_break_lines = (e1_break1_pts, e1_break2_pts, e1_break3_pts)
            else:
                # Ball is on the right side of the court, use Edge 1's break lines
                bounce_break_lines = (e2_break1_pts, e2_break2_pts, e2_break3_pts)

            # Get height category using post-bounce tracking
            height_category = calculate_post_bounce_height(
                bounce_frame, frames, BALL_LOOKUP, front_transform,
                table_min_x, table_max_x, bounce_break_lines
            )

            # If tracking failed, fall back to bounce point calculation
            if height_category is None:
                height_category = get_ball_height_category(bounce_tx, bounce_ty, *bounce_break_lines)

                # Apply height adjustment based on ball's Y position
                # If the ball is very high up in Y coordinate (lower Y values), adjust to ensure "high"
                # If the ball is very low in Y coordinate (higher Y values), adjust to ensure "low"
                if bounce_ty < yt1_warp - 2*step2:  # Far above the highest break line
                    height_category = "high"
                elif bounce_ty > yt1_warp + 50:     # Below the table level
                    height_category = "low"

            # Track the distribution of height categories
            height_counts[height_category] += 1

            # Write to dataset
            data = {
                "bounce_frame": bounce_frame,
                "vx_before": vx_before,
                "vy_before": vy_before,
                "bounce_x": bounce_tx,
                "bounce_y": bounce_ty,
                "height_category": height_category,
            }
            writer.writerow(data)
            print(data)

    # Print the distribution of height categories
    total_bounces = sum(height_counts.values())
    print(f"\nHeight category distribution:")
    for category, count in height_counts.items():
        percentage = (count / total_bounces) * 100 if total_bounces > 0 else 0
        print(f"{category}: {count} ({percentage:.1f}%)")

    print(f"\nDataset created and saved to {dataset_file}")

if __name__ == "__main__":
    main()
