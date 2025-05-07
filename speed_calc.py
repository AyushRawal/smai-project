import json
import math

# Load JSON data
with open("ball_markup_1.json", "r") as file:
    data = json.load(file)

# Convert keys to integers and sort them
frames = sorted(map(int, data.keys()))

# Define real-world scale (update this based on calibration)
PIXELS_PER_METER = 425  # Example: If 50 pixels = 1 meter, adjust accordingly

# Define frame step for speed calculation
FRAME_STEP = 20  # Speed calculated over at least 5 frames

# Calculate speed in m/s over at least 5 frames
speeds = {}
for i in range(len(frames) - FRAME_STEP):
    f1, f2 = frames[i], frames[i + FRAME_STEP]  # Take 5-frame intervals
    x1, y1 = data[str(f1)]["x"], data[str(f1)]["y"]
    x2, y2 = data[str(f2)]["x"], data[str(f2)]["y"]

    # Compute Euclidean distance in pixels
    pixel_distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Convert to meters
    distance_meters = pixel_distance / PIXELS_PER_METER

    # Compute speed (speed = distance / time)
    time_seconds = FRAME_STEP / 120  # Since FPS = 120
    speed_mps = distance_meters / time_seconds

    speeds[f"{f1}->{f2}"] = speed_mps

# Print results
for transition, speed in speeds.items():
    print(f"Speed between frames {transition}: {speed:.2f} m/s")
