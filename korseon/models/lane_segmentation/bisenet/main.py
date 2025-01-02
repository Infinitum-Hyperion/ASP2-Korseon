import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2

import lib.data.transform_cv2 as T
from lib.models import model_factory
from configs import set_cfg_from_file


# uncomment the following line if you want to reduce cpu usage, see issue #231
#  torch.set_num_threads(4)

torch.set_grad_enabled(False)
np.random.seed(123)


# args
parse = argparse.ArgumentParser()
parse.add_argument('--config', dest='config', type=str, default='configs/bisenetv2_city.py',)
parse.add_argument('--weight-path', type=str, default='./weights/model_final_v1_city_new.pth',)
parse.add_argument('--img-path', dest='img_path', type=str, default='./image.png',)
parse.add_argument('--thr', dest='threshold', type=float)
args = parse.parse_args()
cfg = set_cfg_from_file(args.config)


palette = np.random.randint(0, 256, (256, 3), dtype=np.uint8)

# define model
net = model_factory[cfg.model_type](cfg.n_cats, aux_mode='eval')
net.load_state_dict(torch.load(args.weight_path, map_location='cpu'), strict=False)
net.eval()

# prepare data
to_tensor = T.ToTensor(
    mean=(0.3257, 0.3690, 0.3223), # city, rgb
    std=(0.2112, 0.2148, 0.2115),
)
im = cv2.imread(args.img_path)[:, :, ::-1]
im = to_tensor(dict(im=im, lb=None))['im'].unsqueeze(0)

# shape divisor
org_size = im.size()[2:]
new_size = [math.ceil(el / 32) * 32 for el in im.size()[2:]]

# inference
lane_class_id = 7 # road
im = F.interpolate(im, size=new_size, align_corners=False, mode='bilinear')
out_raw = net(im)[0]
out_resized = F.interpolate(out_raw, size=org_size, align_corners=False, mode='bilinear')
out_prob = F.softmax(out_resized, dim=1)  # Shape: (batch_size, n_classes, height, width)
lane_prob = out_prob[:, lane_class_id]  # Shape: (batch_size, height, width)
lane_prob_normalized = (lane_prob - lane_prob.min()) / (lane_prob.max() - lane_prob.min())

# Convert lane probabilities to a numpy array
lane_prob_np = lane_prob.squeeze().cpu().numpy()

# Normalize probabilities
lane_prob_normalized = (lane_prob_np - lane_prob_np.min()) / (lane_prob_np.max() - lane_prob_np.min())

# Compute dynamic threshold using percentile
percentile_threshold = np.percentile(lane_prob_normalized, 95)

# Apply fixed and percentile thresholds
fixed_mask = lane_prob_normalized > args.threshold
percentile_mask = lane_prob_normalized > percentile_threshold

# Combine the masks (logical OR)
lane_mask = np.logical_or(fixed_mask, percentile_mask).astype(np.uint8) * 255

# Spatial Filter to remove building contours
# Assuming `binary_mask` is the current filtered binary lane mask
h, w = lane_mask.shape

# Create a spatial filter that keeps only the bottom half
spatial_filter = np.zeros((h, w), dtype=np.uint8)
spatial_filter[2* h//3:, :] = 1  # Keep bottom half

# Apply spatial filter to the mask
filtered_binary_mask = lane_mask * spatial_filter
cv2.imwrite('./res/sptial_mask.jpg', filtered_binary_mask)

# visualize
# lane_mask = (lane_prob_normalized > args.threshold).squeeze().cpu().numpy().astype(np.uint8) * 255
cv2.imwrite('./res/mask.jpg', lane_mask)


### Morphological Ops

lane_mask = cv2.imread('./res/mask.jpg', cv2.IMREAD_GRAYSCALE)

# Apply morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
cleaned_mask = cv2.morphologyEx(filtered_binary_mask, cv2.MORPH_CLOSE, kernel)  # Fill small gaps
cleaned_mask = cv2.morphologyEx(cleaned_mask, cv2.MORPH_OPEN, kernel)  # Remove noise

# Save the refined mask
cv2.imwrite('./res/cleaned_lane_mask.jpg', cleaned_mask)

### Region Filtering
# Remove small blobs
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned_mask, connectivity=8)
for i in range(1, num_labels):  # Skip background (label 0)
    if stats[i, cv2.CC_STAT_AREA] < 500:  # Adjust the area threshold as needed
        cleaned_mask[labels == i] = 0

# Save the refined mask
cv2.imwrite('./res/filtered_lane_mask.jpg', cleaned_mask)

### Lane Line Approximation
# Apply edge detection
edges = cv2.Canny(cleaned_mask, 50, 150)

# Detect lines using Hough Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=20)

# Draw the detected lines
output_image = cv2.imread('./image.png')  # Load the original image
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Save the output image with lines
cv2.imwrite('./res/lane_lines.jpg', output_image)

