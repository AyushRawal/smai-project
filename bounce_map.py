import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load JSON files
with open("ball_markup_1.json", "r") as f:
    ball_data = json.load(f)

with open("events_markup_1.json", "r") as f:
    events_data = json.load(f)

# Extract (x, y) coordinates for bounce event frames
bounce_points = [
    (ball_data[frame]["x"], ball_data[frame]["y"])
    for frame in events_data if events_data[frame] == "bounce" and frame in ball_data
]

# Convert to NumPy array if there are bounce points
if bounce_points:
    bounce_points = np.array(bounce_points)

    # Plot heat map
    plt.figure(figsize=(10, 6))
    sns.kdeplot(x=bounce_points[:, 0], y=bounce_points[:, 1], cmap="Reds", fill=True, bw_adjust=0.5)
    plt.scatter(bounce_points[:, 0], bounce_points[:, 1], c="blue", s=5, label="Bounce Points")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.title("Bounce Heat Map")
    plt.legend()
    plt.show()
else:
    print("No bounce events found.")
