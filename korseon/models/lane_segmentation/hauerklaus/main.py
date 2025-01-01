import segmentation_models_pytorch as smp
import torch
import cv2
import numpy as np
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2
import os

# Define output directory for saving results
OUTPUT_DIR = "./output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def preprocess_image(image_path):
    """Preprocess the image for the model."""
    # Define preprocessing pipeline
    preprocess = Compose([
        Resize(512, 256),  # Resize to match model's input size
        Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    augmented = preprocess(image=image)
    return augmented["image"].unsqueeze(0)  # Add batch dimension

def predict(image_path, model, device="cpu"):
    """Perform inference on an image."""
    # Preprocess image
    input_tensor = preprocess_image(image_path).to(device)

    # Perform inference
    with torch.no_grad():
        output = model(input_tensor)
        output = torch.sigmoid(output)  # Convert logits to probabilities

    # Convert output to binary mask
    binary_mask = (output.squeeze(0).squeeze(0) > 0.5).cpu().numpy().astype(np.uint8)
    return binary_mask

def save_binary_mask(binary_mask, output_name="binary_mask.png"):
    """Save the binary mask to a file."""
    mask_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(mask_path, binary_mask * 255)  # Convert binary mask to grayscale format
    print(f"Binary mask saved to: {mask_path}")

def overlay_mask(image_path, mask, output_name="lane_overlay.jpg"):
    """Overlay the binary mask onto the original image."""
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask_resized = cv2.resize(mask, (original_image.shape[1], original_image.shape[0]))
    overlay = cv2.addWeighted(original_image, 0.7, cv2.cvtColor(mask_resized * 255, cv2.COLOR_GRAY2BGR), 0.3, 0)

    # Save the overlay
    overlay_path = os.path.join(OUTPUT_DIR, output_name)
    cv2.imwrite(overlay_path, overlay)
    print(f"Lane overlay image saved to: {overlay_path}")
    return overlay

if __name__ == '__main__':
    # Load pretrained UNet model with MobileNetV2 backbone
    model = smp.Unet(
        encoder_name="mobilenet_v2",       # Lightweight backbone
        encoder_weights="imagenet",       # Use pretrained weights
        in_channels=3,                    # RGB images
        classes=1,                        # Binary segmentation (lane vs. background)
    )

    # Set the model to evaluation mode
    model.eval()

    # Example usage
    image_path = "./image.png"  # Path to the test image

    # Perform inference
    binary_mask = predict(image_path, model)

    # Save binary mask separately
    save_binary_mask(binary_mask, output_name="binary_mask.png")

    # Create and save lane overlay
    overlay_mask(image_path, binary_mask, output_name="lane_overlay.jpg")
