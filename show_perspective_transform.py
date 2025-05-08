#!/usr/bin/env python3
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
import json
from pathlib import Path

def perspective_transform_demo(image_path, corners_file=None, output_path=None, interactive=False):
    """
    Demonstrate perspective transformation on an image.
    
    Args:
        image_path: Path to the input image
        corners_file: Optional JSON file with corner coordinates
        output_path: Path to save the output image
        interactive: If True, user can select points interactively
    """
    # Read the image
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Convert BGR to RGB for matplotlib display
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w = image.shape[:2]
    
    # Initialize points
    src_points = None
    
    # If corners file is provided, load points from it
    if corners_file:
        try:
            with open(corners_file, 'r') as f:
                corners = json.load(f)
                src_points = np.float32([
                    [corners["x1_orig"], corners["y1_orig"]],
                    [corners["x2_orig"], corners["y2_orig"]],
                    [corners["x3_orig"], corners["y3_orig"]],
                    [corners["x4_orig"], corners["y4_orig"]]
                ])
                print(f"Loaded corner points from {corners_file}")
        except Exception as e:
            print(f"Error loading corners file: {e}")
            src_points = None
    
    # If no corners file or failed to load, use default or interactive selection
    if src_points is None:
        if interactive:
            # Interactive point selection
            plt.figure(figsize=(10, 8))
            plt.imshow(image_rgb)
            plt.title("Click on 4 points in clockwise order starting from top-left")
            points = plt.ginput(4, timeout=0)
            plt.close()
            src_points = np.float32(points)
            src_points = sorted(src_points, key=lambda x: (x[0], x[1]))
            src_points = np.array([
                src_points[1],
                src_points[2],
                src_points[3],
                src_points[0],
            ], dtype=np.float32)
            print(f"Source points: {src_points}")
        else:
            raise Exception("No corners file provided and interactive mode is not enabled.")


    x1_orig, y1_orig = src_points[0]
    x2_orig, y2_orig = src_points[1]
    x3_orig, y3_orig = src_points[2]
    x4_orig, y4_orig = src_points[3]
    
    avg_inset = ((x1_orig - x4_orig) + (x3_orig - x2_orig)) / 2.0
    xt1_adj = x4_orig + avg_inset
    xt2_adj = x3_orig - avg_inset

    xt3_warp = x3_orig
    xt4_warp = x4_orig

    yt1_warp = yt2_warp = (y1_orig + y2_orig) // 2
    yt3_warp = yt4_warp = (y3_orig + y4_orig) // 2

    # src_points = np.float32(
    #     [[x1_orig, y1_orig], [x2_orig, y2_orig], [x3_orig, y3_orig], [x4_orig, y4_orig]]
    # )
    dst_points = np.float32(
        [
            [xt1_adj, yt1_warp],
            [xt2_adj, yt2_warp],
            [xt3_warp, yt3_warp],
            [xt4_warp, yt4_warp],
        ]
    )
    dst_width, dst_height = 1920, 1080  # Size of the output image

    # Define destination points (rectangle)
    # dst_width, dst_height = 800, 600  # Size of the output image
    # dst_points = np.float32([
    #     [0, 0],                      # Top-left
    #     [dst_width, 0],              # Top-right
    #     [dst_width, dst_height],     # Bottom-right
    #     [0, dst_height]              # Bottom-left
    # ])
    
    # Compute the perspective transform matrix
    transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Apply the perspective transformation
    transformed_image = cv2.warpPerspective(image, transform_matrix, (dst_width, dst_height))
    transformed_image_rgb = cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB)
    
    # Draw the source points on the original image
    image_with_points = image_rgb.copy()
    for i, point in enumerate(src_points):
        cv2.circle(image_with_points, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        cv2.putText(image_with_points, f"P{i+1}", (int(point[0])+10, int(point[1])+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Calculate break lines similar to inference.py
    x1_orig, y1_orig = src_points[0]
    x2_orig, y2_orig = src_points[1]
    x3_orig, y3_orig = src_points[2]
    x4_orig, y4_orig = src_points[3]
    

    
    # Draw the destination points on the transformed image for reference
    transformed_with_points = transformed_image_rgb.copy()
    for i, point in enumerate(dst_points):
        cv2.circle(transformed_with_points, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)
        cv2.putText(transformed_with_points, f"P{i+1}", (int(point[0])+10, int(point[1])+10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # s = yt3_warp - yt2_warp
    # step1 = s * 0.4
    # step2 = s * 0.8
    s = corners["near_net_post_bottom_y"] - corners["near_net_post_top_y"]
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
    
    # Reshape for visualization
    e1_break1_vis = e1_break1_pts.reshape((-1, 1, 2))
    e1_break2_vis = e1_break2_pts.reshape((-1, 1, 2))
    e1_break3_vis = e1_break3_pts.reshape((-1, 1, 2))
    e2_break1_vis = e2_break1_pts.reshape((-1, 1, 2))
    e2_break2_vis = e2_break2_pts.reshape((-1, 1, 2))
    e2_break3_vis = e2_break3_pts.reshape((-1, 1, 2))
    
    # Draw break lines on the original image
    transformed_with_points_and_lines = transformed_with_points.copy()
    cv2.polylines(transformed_with_points_and_lines, [e1_break1_vis], True, (0, 255, 255))
    cv2.polylines(transformed_with_points_and_lines, [e2_break1_vis], True, (0, 255, 255))
    cv2.polylines(transformed_with_points_and_lines, [e1_break2_vis], True, (0, 255, 255))
    cv2.polylines(transformed_with_points_and_lines, [e2_break2_vis], True, (0, 255, 255))
    cv2.polylines(transformed_with_points_and_lines, [e1_break3_vis], True, (0, 255, 255))
    cv2.polylines(transformed_with_points_and_lines, [e2_break3_vis], True, (0, 255, 255))

    # Display the original and transformed images side by side
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    plt.imshow(image_rgb)
    plt.title("Original Image")
    
    plt.subplot(2, 2, 2)
    plt.imshow(image_with_points)
    plt.title("Original Image with Source Points")
    
    plt.subplot(2, 2, 3)
    plt.imshow(transformed_with_points)
    plt.title("Transformed Image")
    
    plt.subplot(2, 2, 4)
    plt.imshow(transformed_with_points_and_lines)
    plt.title("Transformed Image with Destination Points")
    
    plt.tight_layout()
    
    # Save the output if requested
    if output_path:
        plt.savefig(output_path)
        print(f"Output saved to {output_path}")
    
    plt.show()
    
    return transform_matrix

def main():
    parser = argparse.ArgumentParser(description="Demonstrate perspective transformation on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--corners", type=str, help="Path to the corners JSON file (optional)")
    parser.add_argument("--output", type=str, help="Path to save the output image (optional)")
    parser.add_argument("--interactive", action="store_true", help="Enable interactive point selection")
    args = parser.parse_args()
    
    perspective_transform_demo(
        image_path=args.image,
        corners_file=args.corners,
        output_path=args.output,
        interactive=args.interactive
    )

if __name__ == "__main__":
    main()