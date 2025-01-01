import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import glob
from utils.utils import one_hot_encoding

class CULaneDataset(Dataset):
    def __init__(self, root_dir, image_set='train', transforms=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms
        self.image_list, self.lane_list = self._load_data()

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        lane_name = self.lane_list[idx]

        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        lane_mask = self.read_lines_txt(lane_name)

        if self.transforms:
            image, lane_mask = self.transforms(image, lane_mask)

        return image, lane_mask, lane_name  # Return image, lane mask, and lane file name

    def _load_data(self):
        image_list = []
        lane_list = []
        print(f"Dataset root directory: {self.root_dir}") # Debugging print statement

        # Define the specific folders you extracted or want to load
        selected_folders = [
            'driver_100_30frame',
            'driver_161_90frame',
            'driver_193_90frame'
            # Add other driver folders as needed
        ]

        for folder in selected_folders:
            folder_path = os.path.join(self.root_dir, folder)
            print(f"Processing folder: {folder_path}")

            # Use glob to find all .jpg files recursively within the selected folders
            image_paths = glob.glob(os.path.join(folder_path, "*/*.jpg"), recursive=True)
            print(f"Found {len(image_paths)} image paths in {folder_path}")
            
            # Construct corresponding .lines.txt paths
            for image_path in image_paths:
                lane_path = image_path.replace('.jpg', '.lines.txt')
                if os.path.exists(image_path) and os.path.exists(lane_path):
                    image_list.append(image_path)
                    lane_list.append(lane_path)
                else:
                    print(f"Warning: Missing image or lane file for: {image_path}")
        print(f"Total images loaded: {len(image_list)}")

        return image_list, lane_list

    def read_lines_txt(self, lines_txt_path):
        """Reads the .lines.txt file and creates a mask."""
        with open(lines_txt_path, 'r') as f:
            lines = f.readlines()

        # Create an empty mask (adjust dimensions as needed)
        mask = np.zeros((874, 1640), dtype=np.uint8)

        for line in lines:
            points = line.strip().split()
            points = np.array(points, dtype=np.float32).reshape(-1, 2)

            # Draw lines on the mask (adjust thickness, color, etc.)
            cv2.polylines(mask, [points.astype(np.int32)], isClosed=False, color=1, thickness=5)

        return mask