import cv2
import numpy as np
import torch
from model.lanenet import LaneNet
from utils.transforms import Resize, Normalize, ToTensor
from io import BytesIO

# --- Configuration ---
MODEL_PATH = './weights/W2_epoch_1.pth'  # Replace with your trained model path
INPUT_SIZE = (1024, 512)  # Replace with the input size used during training (width, height)
OUTPUT_DIR = './out'  # Directory to save the output masks and images

# Create output directory if it doesn't exist
import os
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Load the Model (CPU) ---
net = LaneNet(num_classes=2, encoder_relu=False, decoder_relu=True)
net.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device('cpu')))
net.eval()

# --- Transforms ---
transforms = Compose([
    Resize(INPUT_SIZE),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensor()
])

def inference_from_png_file(image_path, output_name="output"):
    """
    Performs LaneNet inference on a PNG image file.

    Args:
        image_path: Path to the input PNG image.
        output_name: Base name for output files.
    """

    # --- Load and Preprocess the Image ---
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    original_img = img.copy()

    # Convert to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply transforms
    img, _ = transforms(img, img)

    img = img.unsqueeze(0)  # Add a batch dimension
    img = img.to('cpu')

    # --- Inference ---
    with torch.no_grad():
        binary_seg_output, embedding_output = net(img)

    # --- Postprocessing ---
    # 1. Get Binary Segmentation Map
    binary_seg_output = binary_seg_output.squeeze(0)
    binary_seg_prob = torch.sigmoid(binary_seg_output)
    binary_seg_pred = (binary_seg_prob > 0.5).float()
    binary_seg_img = binary_seg_pred.squeeze(0).cpu().numpy()
    binary_seg_img = (binary_seg_img * 255).astype(np.uint8)

    # 2. Get Instance Segmentation Map (if applicable)
    instance_seg_img = None  # Placeholder

    # 3. Save Binary Mask as JPG
    binary_mask_path = os.path.join(OUTPUT_DIR, f"{output_name}_binm.jpg")
    cv2.imwrite(binary_mask_path, binary_seg_img)
    print(f"Binary segmentation mask saved to: {binary_mask_path}")

    # 4. Save Instance Mask as JPG (if applicable)
    if instance_seg_img is not None:
        instance_mask_path = os.path.join(OUTPUT_DIR, f"{output_name}_inm.jpg")
        cv2.imwrite(instance_mask_path, instance_seg_img)
        print(f"Instance segmentation mask saved to: {instance_mask_path}")

    # 5. Overlay on Original Image
    resized_binary_seg_img = cv2.resize(binary_seg_img, (original_img.shape[1], original_img.shape[0]))
    resized_binary_seg_img = np.stack((resized_binary_seg_img,) * 3, axis=-1)
    resized_binary_seg_img = resized_binary_seg_img.astype(np.uint8)

    lane_overlay = cv2.addWeighted(original_img, 1, resized_binary_seg_img, 0.3, 0)

    # 6. Save Lane Overlay as JPG
    lane_overlay_path = os.path.join(OUTPUT_DIR, f"{output_name}_lane_overlay.jpg")
    cv2.imwrite(lane_overlay_path, lane_overlay)
    print(f"Lane overlay image saved to: {lane_overlay_path}")

if __name__ == '__main__':
    input_image_path = './testimg/image.png'  # Replace with your PNG image path
    inference_from_png_file(input_image_path, 'test_output')