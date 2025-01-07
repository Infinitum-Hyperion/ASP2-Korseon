import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import random

def spatialMask(lane_mask):
  ### Spatial Mask
  # Spatial Filter to remove building contours
  # Assuming `binary_mask` is the current filtered binary lane mask
  h, w = lane_mask.shape

  # Create a spatial filter that keeps only the bottom half
  spatial_filter = np.zeros((h, w), dtype=np.uint8)
  spatial_filter[2* h//3:, :] = 1  # Keep bottom third
  # Apply spatial filter to the mask
  spatial_mask = lane_mask * spatial_filter
  print("Spatial Mask")
  cv2.imwrite('./results/spatial_mask.jpg', spatial_mask)
  return spatial_mask

def cleanMask(spatial_mask):
  ### Clean Mask
  # Apply morphological operations
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
  cleaned_mask = cv2.morphologyEx(spatial_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
  cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)  # Remove noise

  # Save the refined mask
  print("Cleaned Mask")
  cv2.imwrite('./results/cleaned_mask.jpg', cleaned_mask)
  return cleaned_mask

def linesMask(spatial_mask):
  lines = cv2.HoughLinesP(
      spatial_mask,
      rho=1,
      theta=np.pi / 180,
      threshold=50,  # Adjust based on input image
      minLineLength=30,  # Minimum length of line segments
      maxLineGap=20,  # Maximum gap to consider segments connected
  )

  # Overlay detected lines on the original mask
  lines_mask = np.zeros_like(spatial_mask)
  if lines is not None:
      for line in lines:
          x1, y1, x2, y2 = line[0]
          cv2.line(lines_mask, (x1, y1), (x2, y2), 255, thickness=2)
  print("Lines Mask")
  # cv2.imshow(lines_mask)
  return lines_mask

def findContours(lines_mask, lane_mask):
  # Find contours in the skeletonized lane mask
  contours, _ = cv2.findContours(lines_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

  # --- Visualization: Draw ALL Contours ---
  all_contours_canvas = cv2.cvtColor(lines_mask, cv2.COLOR_GRAY2BGR)  # Create a color canvas
  cv2.drawContours(all_contours_canvas, contours, -1, (0, 255, 0), 2)  # Draw all contours in green
  # cv2.imshow(all_contours_canvas)

  # Initial filtering
  initial_min_area = 200
  initial_filtered_contours = [contour for contour in contours if cv2.contourArea(contour) > initial_min_area]

  initial_filtered_canvas = cv2.cvtColor(lines_mask, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(initial_filtered_canvas, initial_filtered_contours, -1, (0, 204, 255), 2)
  # cv2.imshow(initial_filtered_canvas)

  # Secondary filtering (lower threshold for bottom regions)
  secondary_min_area = 300
  y_threshold = lane_mask.shape[0] * 0.7  # Adjust this threshold (e.g., 0.5, 0.7)
  filtered_contours = []
  for contour in initial_filtered_contours:
      moments = cv2.moments(contour)
      if moments["m00"] != 0:
          cy = int(moments["m01"] / moments["m00"])
          if cy > y_threshold:  # Check if centroid is in the bottom part
              if cv2.contourArea(contour) > secondary_min_area:
                  filtered_contours.append(contour)
          else:
              filtered_contours.append(contour)
  f2_can = cv2.cvtColor(lines_mask, cv2.COLOR_GRAY2BGR)
  cv2.drawContours(f2_can, filtered_contours, -1, (0, 0, 255), 2)
  print("Initial filtered contours")
  # cv2.imshow(f2_can)
  return filtered_contours

def runDBSCAN(filtered_contours, lines_mask, imgPath):
  # Calculate centroids of contours
  centroids = []
  for contour in filtered_contours:
      moments = cv2.moments(contour)
      if moments["m00"] != 0:  # Avoid division by zero
          cx = int(moments["m10"] / moments["m00"])
          cy = int(moments["m01"] / moments["m00"])
          centroids.append([cx, cy])
  centroids = np.array(centroids)

  canvas_width, canvas_height = 2560, 1370
  if np.any(centroids > max(canvas_width, canvas_height)):
      centroids = centroids * [canvas_width / np.max(centroids[:, 0]), canvas_height / np.max(centroids[:, 1])]

  # Apply DBSCAN clustering
  dbscan = DBSCAN(eps=20, min_samples=1).fit(centroids)  # Adjust eps based on scale
  labels = dbscan.labels_

  dbscan_canvas = cv2.cvtColor(lines_mask, cv2.COLOR_GRAY2BGR)  # Use lane_mask here
  cluster_colors = {
      label: [random.randint(0, 255) for _ in range(3)] for label in set(labels)
  }
  cluster_colors[-1] = [0, 0, 255]  # Red for noise
  for i, label in enumerate(labels):
      color = cluster_colors[label]
      cv2.polylines(dbscan_canvas, [filtered_contours[i]], isClosed=False, color=color, thickness=2)
  # cv2.imshow(dbscan_canvas)

  # Count points in each cluster
  unique_labels, counts = np.unique(labels, return_counts=True)
  cluster_counts = dict(zip(unique_labels, counts))

  # Filter out small clusters
  min_cluster_size = 1  # Example threshold
  filtered_labels = [label if cluster_counts[label] >= min_cluster_size else -1 for label in labels]

  # Post scan filtering
  post_dbscan_min_area = 4500

  # Calculate cluster areas
  cluster_areas = {} # (filtered_contours_INDEX, label)
  for i, label in enumerate(filtered_labels):
      if label != -1:
          if label not in cluster_areas:
              cluster_areas[label] = 0
          cluster_areas[label] += cv2.contourArea(filtered_contours[i])

  filtered_labels_by_area = [
      label if (label == -1 or cluster_areas[label] >= post_dbscan_min_area)
      else -1
      for label in filtered_labels
  ]


  # Generate a blank canvas for visualization
  canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

  # Assign random colors to each cluster
  cluster_colors = {label: [random.randint(0, 255) for _ in range(3)] for label in set(filtered_labels_by_area)}  # label: [B, G, R]
  cluster_colors[-1] = [0, 0, 255]  # Red for noise

  # Draw the centroids and contours for each cluster
  for idx, label in enumerate(filtered_labels_by_area):
      color = cluster_colors[label]
      centroid = tuple(centroids[idx])
      if 0 <= centroid[0] < canvas.shape[1] and 0 <= centroid[1] < canvas.shape[0]:
          cv2.circle(canvas, centroid, 5, color, -1)  # Draw centroid
          cv2.polylines(canvas, [filtered_contours[idx]], isClosed=False, color=color, thickness=2)  # Draw contour
      else:
          print(f"Centroid {centroid} is out of bounds.")

  for idx, contour in enumerate(filtered_contours):
      x, y, w, h = cv2.boundingRect(contour)
      cv2.rectangle(canvas, (x, y), (x + w, y + h), cluster_colors[filtered_labels_by_area[idx]], 1)

  # Save and display the result
  # cv2.imwrite("clusters_visualization.jpg", canvas)
  print("DBSCAN Processed Result")
  cv2.imwrite('./results/dbscan_result.jpg', canvas)
  # Load original image (replace 'original_image_path' with your image path)
  original_image = cv2.imread(imgPath)

  alpha = 0.5  # Transparency factor
  overlay = cv2.addWeighted(original_image, 1 - alpha, canvas, alpha, 0)

  # Display the result
  print("Overlaid Result")
  cv2.imwrite('./results/dbscan_overlay.jpg', overlay)
  return filtered_labels_by_area, cluster_colors, filtered_contours, canvas

def fitPolynomials(filtered_labels_by_area, cluster_colors, filtered_contours, imgPath):
  # Fit polynomials to each contour and draw on the original image
  coeffs = {}
  linspaces = {}
  original_image = cv2.imread(imgPath)
  for idx, label in enumerate(filtered_labels_by_area):
      if label != -1:  # Skip noise and filtered clusters
          color = cluster_colors[label]
          contour = filtered_contours[idx]

          # Extract x and y coordinates from the contour
          x = contour[:, 0, 0]
          y = contour[:, 0, 1]

          # Fit a polynomial (e.g., 2nd degree)
          try:
              poly_coeffs = np.polyfit(x, y, 2)  # Fit a 2nd-degree polynomial
              coeffs[idx] = poly_coeffs
              # Create a polynomial function
              poly_func = np.poly1d(poly_coeffs)

              # Generate points along the fitted polynomial
              x_min = min(x)
              x_max = max(x)
              linspaces[idx] = (x_min, x_max)
              x_fit = np.linspace(x_min, x_max, 100)  # Generate 100 points for smooth curve
              y_fit = poly_func(x_fit)

              # Draw the fitted polynomial on the original image
              points = np.array([x_fit, y_fit], np.int32).T  # Convert to (x, y) pairs
              points = points.reshape((-1, 1, 2)) # Reshape for cv2.polylines
              cv2.polylines(original_image, [points], isClosed=False, color=color, thickness=2)

          except np.RankWarning:
              print(f"RankWarning: Polyfit may be poorly conditioned for contour {idx}")
              # Handle the warning (e.g., by drawing the original contour or skipping)
              cv2.polylines(original_image, [contour], isClosed=False, color=color, thickness=2)
          except Exception as e:
              print(f"An error occurred for contour {idx}: {e}")
              cv2.polylines(original_image, [contour], isClosed=False, color=color, thickness=2)

  # Display the result
  print("Polynomials Detected")
  cv2.imwrite('./results/polynomials.jpg', original_image)
  return coeffs, linspaces