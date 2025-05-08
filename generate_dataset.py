import argparse  # Import argparse for command line argument parsing
import csv
import json
from pathlib import Path

import cv2  # Import cv2 at the top level
import numpy as np

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
    """
    Find which interval a point (x0, y0) is in relative to three lines.
    Lines are defined by ax + by + c = 0.
    Assumes c_values are ordered: c_values[0] for the lowest line (L0),
                                 c_values[1] for the middle line (L1),
                                 c_values[2] for the highest line (L2).
    Assumes that the expression (a*x0 + b*y0 + c) < 0 means the point (x0,y0)
    is physically HIGHER than the line.
    """
    if len(c_values) != 3:
        # It's better to raise a specific error type
        raise ValueError("Error: Exactly three c values must be provided.")

    epsilon = 1e-9  # Tolerance for floating-point comparisons
    vals = [a * x0 + b * y0 + c for c in c_values]

    # vals[0] corresponds to L0 (e.g., net height - lowest physical threshold)
    # vals[1] corresponds to L1 (middle break line threshold)
    # vals[2] corresponds to L2 (e.g., high break - highest physical threshold)

    # Decision logic based on the assumption: val < 0 means point is physically HIGHER.

    # Case 1: Point is physically higher than the highest line (L2)
    if vals[2] < -epsilon:
        return "high"
    # Case 2: Point is on the highest line (L2)
    elif abs(vals[2]) < epsilon:
        # Convention: on the boundary, classify as the higher category or specific.
        # Here, we can say it meets the "high" threshold boundary.
        return "high"
    # Case 3: Point is below L2 (vals[2] > epsilon theoretically)
    # AND physically higher than the middle line (L1)
    elif vals[1] < -epsilon:
        return "mid"
    # Case 4: Point is on the middle line (L1)
    elif abs(vals[1]) < epsilon:
        return "mid" # On the boundary of mid/low, part of "mid" region
    # Case 5: Point is below L1 (vals[1] > epsilon theoretically)
    # AND physically higher than the lowest line (L0)
    elif vals[0] < -epsilon:
        return "low" # Region between net and mid-break
    # Case 6: Point is on the lowest line (L0)
    elif abs(vals[0]) < epsilon:
        return "low" # On the net
    # Case 7: Point is physically lower than all lines (including L0)
    # This means vals[0], vals[1], and vals[2] are all > epsilon
    else:
        return "low"

def find_line_interval_from_points(lines_points, x0, y0):
    """
    Find which interval a point is in based on line points.
    Normalizes the line equation so that the 'b' coefficient (for y)
    is non-negative, ensuring consistent behavior in find_line_interval.
    """
    if len(lines_points) != 3:
        raise ValueError("Error: Exactly three lines must be provided.") # Good practice for specific errors

    # Determine 'a' and 'b' from the first line in the set.
    # These define the slope and orientation, assumed consistent for parallel lines.
    point1_line1 = lines_points[0][0]
    point2_line1 = lines_points[0][1]
    x1_l1, y1_l1 = point1_line1[0], point1_line1[1]
    x2_l1, y2_l1 = point2_line1[0], point2_line1[1]

    # Original 'a' and 'b' based on the code's convention:
    # ax + by + c = 0 where a=dy, b=-dx
    # So the term with y is (-dx) * y
    a_orig = y2_l1 - y1_l1  # dy
    b_orig = -(x2_l1 - x1_l1) # -dx (this is the coefficient for y0 in ax+by+c)

    final_a = a_orig
    final_b = b_orig
    sign_flipper = 1.0

    # We want final_b (coefficient of y) to be positive, so that
    # in 'val = final_a*x0 + final_b*y0 + c_final', a smaller y0 (higher ball)
    # leads to a smaller 'val'.
    if final_b < 0:
        sign_flipper = -1.0
        final_a = a_orig * sign_flipper
        final_b = b_orig * sign_flipper # This will now be positive

    # Calculate c_values using the (potentially flipped) final_a and final_b
    # c = -(ax_pt + by_pt)
    # Each line in lines_points has its own intercept, but shares the orientation (final_a, final_b)
    final_c_values = []
    for line_pair_pts in lines_points:
        pt_on_line_x, pt_on_line_y = line_pair_pts[0][0], line_pair_pts[0][1] # Use first point of each line
        c_val = -(final_a * pt_on_line_x + final_b * pt_on_line_y)
        # If we flipped a and b, c effectively gets multiplied by sign_flipper as well.
        # However, the above calculation using final_a, final_b directly gives the correct c_val.
        final_c_values.append(c_val)

    return find_line_interval(final_a, final_b, final_c_values, x0, y0)


def get_ball_height_category(tx, ty, break_line1_pts, break_line2_pts, break_line3_pts):
    """Determine ball height category"""
    lines = [
        np.array(break_line1_pts, dtype=np.float32),
        np.array(break_line2_pts, dtype=np.float32),
        np.array(break_line3_pts, dtype=np.float32),
    ]
    return find_line_interval_from_points(lines, float(tx), float(ty))


def calculate_post_bounce_height(
    bounce_frame, frames, BALL_LOOKUP, front_transform, x_min, x_max, break_lines
):
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
                    if (last_velocity_y * current_velocity_y < 0) and abs(
                        current_velocity_y
                    ) > 1.0:
                        velocity_direction_changed = True
                        break

                last_velocity_y = current_velocity_y

            # Check if ball reached table edge
            if tx <= x_min or tx >= x_max:
                break

            # Check if we have enough points to make a determination
            if len(positions) >= 3 and (
                velocity_direction_changed or len(positions) >= 10
            ):
                # If we've collected enough data points, we can make a determination
                break

    if len(positions) < 2:  # Need at least 2 points for analysis
        return None

    # Find the highest point (minimum y value)
    min_height = float("inf")  # Track minimum height (maximum y position)
    min_height_index = 0

    for i in range(len(positions)):
        if heights[i] < min_height:
            min_height = heights[i]
            min_height_index = i

    # Use the minimum height point as our key position for categorization
    best_position = positions[min_height_index]

    # Determine height category using the best position
    height_category = get_ball_height_category(
        best_position[0], best_position[1], *break_lines
    )

    return height_category


def main():
    # Argument parser for command line arguments
    parser = argparse.ArgumentParser(description="Process ball bounce data.")
    parser.add_argument(
        "--corners",
        type=str,
        default="corners.json",
        help="Path to the corners coordinates JSON file",
    )
    parser.add_argument(
        "--ball_markup",
        type=str,
        default="ball_markup.json",
        help="Path to the ball markup JSON file",
    )
    parser.add_argument(
        "--event_markup",
        type=str,
        default="events_markup.json",
        help="Path to the events markup JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="ball_bounce_dataset_from_json.csv",
        help="Output CSV file for the dataset",
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
    # s = yt3_warp - yt2_warp
    # step1 = s * 0.4
    # step2 = s * 1.2

    near_net_post_top_transform = transform_ball_position(
        CORNERS["near_net_post_top_x"],
        CORNERS["near_net_post_top_y"],
        front_transform,
    )
    near_net_post_bottom_transform = transform_ball_position(
        CORNERS["near_net_post_bottom_x"],
        CORNERS["near_net_post_bottom_y"],
        front_transform,
    )

    s = near_net_post_bottom_transform[1] - near_net_post_top_transform[1]
    step1 = s * 1.2
    step2 = s * 3

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
    mid_court_x = (
        max(x_range_edge1_side[1], x_range_edge2_side[1])
        + min(x_range_edge1_side[0], x_range_edge2_side[0])
    ) / 2

    # Define the table boundaries
    table_min_x = min(xt4_warp, xt1_adj, xt2_adj, xt3_warp)
    table_max_x = max(xt4_warp, xt1_adj, xt2_adj, xt3_warp)

    # Convert string frame indices to integers for sorting
    frames = [
        int(frame) for frame in BALL_LOOKUP.keys() if BALL_LOOKUP[frame]["x"] != -1
    ]
    frames.sort()  # Ensure frames are in order

    # Find bounce frames
    bounce_frames = [
        int(frame) for frame in EVENT_LOOKUP.keys() if EVENT_LOOKUP[frame] == "bounce"
    ]
    bounce_frames.sort()

    # Track height category distribution
    height_counts = {"low": 0, "mid": 0, "high": 0, "N/A": 0}

    # Prepare dataset
    dataset_file = Path(args.output)
    with dataset_file.open("w", newline="") as csvfile:
        fieldnames = [
            "bounce_frame",
            "tx1",
            "ty1",
            "tx2",
            "ty2",
            "tx3",
            "ty3",
            "tx4",
            "ty4",
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
            if (
                bounce_frame_str not in BALL_LOOKUP
                or BALL_LOOKUP[bounce_frame_str]["x"] == -1
            ):
                print(f"No valid ball position for bounce frame {bounce_frame}")
                continue

            bounce_x = BALL_LOOKUP[bounce_frame_str]["x"]
            bounce_y = BALL_LOOKUP[bounce_frame_str]["y"]

            # Transform ball position
            bounce_tx, bounce_ty = transform_ball_position(
                bounce_x, bounce_y, front_transform
            )

            # Get frames before bounce for velocity calculation
            frame_window = 60  # Same as in original script
            pre_bounce_frames = [
                f for f in frames if bounce_frame - frame_window <= f < bounce_frame
            ]

            if len(pre_bounce_frames) < 2:
                print(
                    f"Not enough frames before bounce at frame {bounce_frame} for velocity calculation"
                )
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
                bounce_frame,
                frames,
                BALL_LOOKUP,
                front_transform,
                table_min_x,
                table_max_x,
                bounce_break_lines,
            )

            # If tracking failed, fall back to bounce point calculation
            if height_category is None:
                height_category = get_ball_height_category(
                    bounce_tx, bounce_ty, *bounce_break_lines
                )

                # Apply height adjustment based on ball's Y position
                # If the ball is very high up in Y coordinate (lower Y values), adjust to ensure "high"
                # If the ball is very low in Y coordinate (higher Y values), adjust to ensure "low"
                # if bounce_ty < yt1_warp - 2 * step2:  # Far above the highest break line
                #     height_category = "high"
                # elif bounce_ty > yt1_warp + 50:  # Below the table level
                #     height_category = "low"

            # Track the distribution of height categories
            height_counts[height_category] += 1

            # Write to dataset

            data = {
                "bounce_frame": bounce_frame,
                "tx1": x1_orig,
                "ty1": y1_orig,
                "tx2": x2_orig,
                "ty2": y2_orig,
                "tx3": x3_orig,
                "ty3": y3_orig,
                "tx4": x4_orig,
                "ty4": y4_orig,
                "vx_before": vx_before,
                "vy_before": vy_before,
                "bounce_x": bounce_x,
                "bounce_y": bounce_y,
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
