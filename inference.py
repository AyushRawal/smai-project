#!/usr/bin/env python3
"""
Display height categories on video frames.
This script takes an input CSV with frame info and a video file, then
displays the height category for each frame.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm


# Define the HeightNet model (same as in model.py)
class HeightNet(nn.Module):
    def __init__(self, in_dim=12, num_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def load_model(model_path):
    """Load the trained HeightNet model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HeightNet().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def get_scaler(train_csv_path):
    """Create a StandardScaler fitted on training data"""
    train_df = pd.read_csv(train_csv_path)
    X_train = train_df[
        [
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
        ]
    ].values.astype("float32")
    scaler = StandardScaler().fit(X_train)
    return scaler


def predict_height_category(model, scaler, sample_raw, device):
    """Predict height category from raw input features"""
    label_map = {0: "low", 1: "medium", 2: "high"}
    sample_std = scaler.transform(sample_raw.astype("float32").reshape(1, -1))
    with torch.no_grad():
        x = torch.from_numpy(sample_std).to(device)
        probs = torch.softmax(model(x), dim=1)
        label_idx = torch.argmax(probs, dim=1).item()
    return label_map[label_idx], probs.cpu().numpy()


def process_video(
    input_csv, video_path, model, scaler, device, corners, output_path=None
):
    """Process video and display height categories on frames"""
    # Load CSV data with frame information
    df = pd.read_csv(input_csv)

    # Map frame numbers to height categories
    frame_to_category = {}
    frame_to_true_category = {}
    category_colors = {
        "low": (0, 0, 255),  # Red
        "medium": (0, 255, 255),  # Yellow
        "high": (0, 255, 0),  # Green
    }

    for idx, row in df.iterrows():
        if "bounce_frame" in df.columns:
            frame_num = int(row["bounce_frame"])
            features = np.array(
                [
                    row["tx1"],
                    row["ty1"],
                    row["tx2"],
                    row["ty2"],
                    row["tx3"],
                    row["ty3"],
                    row["tx4"],
                    row["ty4"],
                    row["vx_before"],
                    row["vy_before"],
                    row["bounce_x"],
                    row["bounce_y"],
                ]
            )
            category, _ = predict_height_category(model, scaler, features, device)
            category_true = row["height_category"]
            frame_to_category[frame_num] = category
            frame_to_true_category[frame_num] = category_true
            
    x1_orig, y1_orig = corners["x1_orig"], corners["y1_orig"]
    x2_orig, y2_orig = corners["x2_orig"], corners["y2_orig"]
    x3_orig, y3_orig = corners["x3_orig"], corners["y3_orig"]
    x4_orig, y4_orig = corners["x4_orig"], corners["y4_orig"]

    s = corners["near_net_post_bottom_y"] - corners["near_net_post_top_y"]
    step1 = s * 1.2
    step2 = s * 3
    # step3 = 180
    e1_break1_pts = np.array([[x1_orig, y1_orig], [x4_orig, y4_orig]], np.int32)
    e1_break2_pts = np.array(
        [[x1_orig, y1_orig - step1], [x4_orig, y4_orig - step1]], np.int32
    )
    e1_break3_pts = np.array(
        [[x1_orig, y1_orig - step2], [x4_orig, y4_orig - step2]], np.int32
    )
    # e1_break3_pts = np.array([[x1_orig, y1_orig - step3], [x4_orig, y4_orig - step3]], np.int32)

    e2_break1_pts = np.array([[x2_orig, y2_orig], [x3_orig, y3_orig]], np.int32)
    e2_break2_pts = np.array(
        [[x2_orig, y2_orig - step1], [x3_orig, y3_orig - step1]], np.int32
    )
    e2_break3_pts = np.array(
        [[x2_orig, y2_orig - step2], [x3_orig, y3_orig - step2]], np.int32
    )
    e1_break1_vis = e1_break1_pts.reshape((-1, 1, 2))
    e1_break2_vis = e1_break2_pts.reshape((-1, 1, 2))
    e1_break3_vis = e1_break3_pts.reshape((-1, 1, 2))
    e2_break1_vis = e2_break1_pts.reshape((-1, 1, 2))
    e2_break2_vis = e2_break2_pts.reshape((-1, 1, 2))
    e2_break3_vis = e2_break3_pts.reshape((-1, 1, 2))

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create output video writer if needed
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Process each frame
    current_frame = 0
    category_display_time = 30  # Show category for 30 frames
    last_category_frame = -1
    current_category = None

    # Get total frames for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Processing {total_frames} frames...")
    for _ in tqdm(range(total_frames)):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.polylines(frame, [e1_break1_vis], True, (0, 255, 255))
        cv2.polylines(frame, [e2_break1_vis], True, (0, 255, 255))
        cv2.polylines(frame, [e1_break2_vis], True, (0, 255, 255))
        cv2.polylines(frame, [e2_break2_vis], True, (0, 255, 255))
        cv2.polylines(frame, [e1_break3_vis], True, (0, 255, 255))
        cv2.polylines(frame, [e2_break3_vis], True, (0, 255, 255))

        # Check if we have height category info for this frame
        if current_frame in frame_to_category:
            current_category = frame_to_category[current_frame]
            current_category_true = frame_to_true_category[current_frame]
            last_category_frame = current_frame

        # Display category if we're within display window of a categorized frame
        if (
            current_category
            and (current_frame - last_category_frame) < category_display_time
        ):
            color = category_colors[current_category]
            cv2.putText(
                frame,
                f"Height: {current_category.upper()}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )
            # show true height category for 2 seconds
            cv2.putText(
                frame,
                f"True Height: {current_category_true.upper()}",
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
                cv2.LINE_AA,
            )
            cv2.waitKey(2000 // 120)

        if out:
            out.write(frame)
        else:
            cv2.imshow("Frame with Height Category", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

        current_frame += 1

    # Clean up
    cap.release()
    if out:
        out.close()
    cv2.destroyAllWindows()

    if output_path:
        print(f"Output video saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Display height categories on video frames"
    )
    parser.add_argument(
        "input_csv", help="Path to the CSV file with frame and bounce data"
    )
    parser.add_argument("video_path", help="Path to the video file")
    parser.add_argument(
        "corners_json", help="Path to the corners.json corresponding to the video file"
    )
    parser.add_argument(
        "--model", default="height_net_best.pt", help="Path to the trained model"
    )
    parser.add_argument(
        "--train_csv", default="dataset.csv", help="Path to the training CSV for scaler"
    )
    parser.add_argument("--output", help="Path to save the output video (optional)")
    args = parser.parse_args()

    # Load model and scaler
    model, device = load_model(args.model)
    scaler = get_scaler(args.train_csv)
    with Path(args.corners_json).open() as f:
        corners = json.load(f)

    # Process the video
    process_video(
        args.input_csv, args.video_path, model, scaler, device, corners, args.output
    )


if __name__ == "__main__":
    main()
