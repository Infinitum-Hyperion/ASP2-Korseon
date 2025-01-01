import os
import shutil
from tqdm import tqdm
import argparse
from PIL import Image, ImageDraw
import numpy as np

def create_dirs(base_dir):
    """Creates necessary directories within the base directory."""
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'masks'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'lists'), exist_ok=True)

def create_symlinks(src_dir, dst_dir, file_exts=['.jpg', '.png']):
    """Creates symbolic links from src_dir to dst_dir for files with specified extensions."""
    print(f"create_symlinks: Creating symbolic links from {src_dir} to {dst_dir}")
    for root, _, files in os.walk(src_dir):
        for file in files:
            if any(file.endswith(ext) for ext in file_exts):
                src_path = os.path.join(root, file)
                rel_path = os.path.relpath(src_path, src_dir)  # Path relative to driver_100_30frame
                dst_path = os.path.join(dst_dir, rel_path)

                print(f"create_symlinks:    Processing file: {file}")
                print(f"create_symlinks:      Source path: {src_path}")
                print(f"create_symlinks:      Relative path: {rel_path}")
                print(f"create_symlinks:      Destination path: {dst_path}")

                os.makedirs(os.path.dirname(dst_path), exist_ok=True)

                if os.path.exists(dst_path):
                    if os.path.islink(dst_path):
                        print(f"create_symlinks:      Removing existing symlink: {dst_path}")
                        os.remove(dst_path)  # Remove existing symlink
                    else:
                        print(f"create_symlinks:      Error: Cannot create symlink. Regular file exists: {dst_path}")
                        continue  # Skip to the next file

                try:
                    print(f"create_symlinks:      Creating symlink: {src_path} -> {dst_path}")
                    os.symlink(src_path, dst_path)
                except OSError as e:
                    print(f"create_symlinks:      Error creating symbolic link: {e}")
                    print(f"create_symlinks:        Source: {src_path}")
                    print(f"create_symlinks:        Destination: {dst_path}")

def generate_instance_masks(culane_root, data_dir, list_file):
    """Generates instance masks from CULane annotations."""
    with open(os.path.join(data_dir, 'lists', list_file), 'r') as f:
        lines = f.readlines()

    for line in tqdm(lines, desc="Generating instance masks"):
        img_rel_path = line.strip() + '.jpg'
        mask_rel_path = line.strip() + '.png'
        img_path = os.path.join(data_dir, 'images', img_rel_path)
        mask_path = os.path.join(data_dir, 'masks', mask_rel_path)

        print(f"generate_instance_masks: Processing image: {img_path}")
        print(f"generate_instance_masks: Mask path: {mask_path}")

        if not os.path.exists(img_path):
            print(f"generate_instance_masks: Error: Image file not found: {img_path}")
            continue

        try:
            img = Image.open(img_path)
        except FileNotFoundError as e:
            print(f"generate_instance_masks: Error opening image: {e}")
            print(f"generate_instance_masks:   Image path: {img_path}")
            continue
        except Exception as e:
            print(f"generate_instance_masks: Error processing image: {e}")
            continue

        w, h = img.size
        mask = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(mask)

        # Construct path to .lines.txt file
        lines_file_rel_path = line.strip() + '.lines.txt'
        lines_file_path = os.path.join(culane_root, 'driver_100_30frame', lines_file_rel_path)

        if not os.path.exists(lines_file_path):
            print(f"generate_instance_masks: Error: Annotation file not found: {lines_file_path}")
            continue

        with open(lines_file_path, 'r') as lf:
            idx = 1
            for lane_line in lf:
                coordinates = lane_line.strip().split()
                if len(coordinates) >= 4:
                    lane_coords = []
                    for i in range(0, len(coordinates), 2):
                        x, y = int(round(float(coordinates[i]))), int(round(float(coordinates[i + 1])))
                        lane_coords.append((x, y))

                    draw.line(lane_coords, fill=idx, width=10)
                    idx += 1

        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        mask.save(mask_path)

def generate_list_files(culane_root, output_dir):
    """Generates train.txt and val.txt files based on CULane structure."""
    train_list_file = os.path.join(output_dir, 'lists', 'train.txt')
    val_list_file = os.path.join(output_dir, 'lists', 'val.txt')

    print(f"generate_list_files: train_list_file = {train_list_file}")
    print(f"generate_list_files: val_list_file = {val_list_file}")

    train_images = []
    for root, _, files in os.walk(os.path.join(culane_root, "driver_100_30frame")):
        for file in files:
            if file.endswith(".jpg"):
                # Get the relative path from the driver_100_30frame directory
                relpath = os.path.relpath(os.path.join(root, file), os.path.join(culane_root, "driver_100_30frame"))
                train_images.append(relpath[:-4])

    print(f"generate_list_files: Found {len(train_images)} training images")

    # Shuffle and split into train and val
    np.random.shuffle(train_images)
    split_idx = int(0.9 * len(train_images))
    train_data = train_images[:split_idx]
    val_data = train_images[split_idx:]

    print(f"generate_list_files: train_data has {len(train_data)} images")
    print(f"generate_list_files: val_data has {len(val_data)} images")

    # Write to train.txt and val.txt
    with open(train_list_file, "w") as f:
        for item in train_data:
            print(f"generate_list_files: Writing to train.txt: {item}")
            f.write(f"{item}\n")

    with open(val_list_file, "w") as f:
        for item in val_data:
            print(f"generate_list_files: Writing to val.txt: {item}")
            f.write(f"{item}\n")

def main():
    parser = argparse.ArgumentParser(description="Prepare CULane dataset for instance segmentation.")
    parser.add_argument("--culane_root", required=True, help="Root directory of the CULane dataset")
    parser.add_argument("--output_dir", default="./CULane_prepared", help="Output directory for prepared dataset")
    args = parser.parse_args()

    print("Creating directories...")
    create_dirs(args.output_dir)

    print("Generating list files...")
    generate_list_files(args.culane_root, args.output_dir)

    print("Creating symbolic links for images...")
    img_dir = os.path.join(args.output_dir, "images")
    create_symlinks(os.path.join(args.culane_root, "driver_100_30frame"), img_dir, ['.jpg'])

    print("Generating instance masks...")
    generate_instance_masks(args.culane_root, args.output_dir, "train.txt")
    generate_instance_masks(args.culane_root, args.output_dir, "val.txt")

    print("CULane dataset preparation complete.")

if __name__ == "__main__":
    main()