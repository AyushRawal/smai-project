import json
import numpy as np

# Load JSON files
with open("ball_markup_1.json", "r") as f:
    ball_data = json.load(f)

with open("events_markup_1.json", "r") as f:
    events_data = json.load(f)

# Convert keys to integers for proper sorting
ball_data = {int(frame): coords for frame, coords in ball_data.items()}
events_data = {int(frame): event for frame, event in events_data.items()}

# Sort frames
sorted_frames = sorted(ball_data.keys())

# Function to calculate speed between two frames
def calculate_speed(frame1, frame2):
    if frame1 in ball_data and frame2 in ball_data:
        x1, y1 = ball_data[frame1]["x"], ball_data[frame1]["y"]
        x2, y2 = ball_data[frame2]["x"], ball_data[frame2]["y"]
        return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)  # Euclidean distance
    return None

# Function to compute average speed over a window of N frames
def average_speed(center_frame, offset):
    speeds = []
    for i in range(1, offset + 1):  # Take `offset` frames before/after
        speed = calculate_speed(center_frame - i, center_frame - (i - 1))
        if speed:
            speeds.append(speed)
    return np.mean(speeds) if speeds else None

# Store speed drop calculations
speed_drops = {}

window_size = 1 # Look at 5 frames before and after the bounce

for bounce_frame in sorted(events_data.keys()):
    if events_data[bounce_frame] == "bounce":
        speed_before = average_speed(bounce_frame, -window_size)
        speed_after = average_speed(bounce_frame, window_size)

        if speed_before and speed_after and speed_before > 0:
            drop_percentage = ((speed_before - speed_after) / speed_before) * 100
            speed_drops[bounce_frame] = (speed_before, speed_after, drop_percentage)

# Output results
for frame, (before, after, drop) in speed_drops.items():
    print(f"Bounce at frame {frame}: Avg Speed Before = {before:.2f}, Avg Speed After = {after:.2f}, Speed Drop = {drop:.2f}%")
