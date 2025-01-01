import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.lanenet import LaneNet
from model.enet import ENet
from dataset.culane import CULaneDataset  # Or your dataset loader (BDD100K, etc.)
from utils.loss import BinaryFocalLoss, DiscriminativeLoss
from utils.transforms import *
from utils.utils import one_hot_encoding
import time

GEN = 'Wickelheim'
VERSION = 'W2'
EPOCHS = 4
CHECKPOINTS_DIR = f'./drive/MyDrive/ColabNotebooks/{GEN}/{VERSION}/checkpoints'

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="culane", help="Dataset to use (culane, bdd100k, etc.)")
    parser.add_argument("--data_dir", type=str, default="./data/culane", required=True, help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of data loader workers")
    parser.add_argument("--save_dir", type=str, default=CHECKPOINTS_DIR, help="Directory to save checkpoints")
    parser.add_argument("--img_size", type=int, default=512, help="height of resized image")
    parser.add_argument("--img_width", type=int, default=896, help="width of resized image")
    parser.add_argument("--pretrained_weights", type=str, help="Path to pretrained ENet weights (optional)")
    # Add more arguments as needed (e.g., for loss function parameters)
    return parser.parse_args()

def train(model, train_loader, criterion_seg, criterion_instance, optimizer, epoch, device):
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_instance_loss = 0
    for batch_idx, (images, seg_labels, lane_names) in enumerate(train_loader):
        # Move data to the GPU
        images, seg_labels = images.to(device), seg_labels.to(device)

        optimizer.zero_grad()

        binary_seg_output, embedding_output = model(images)
        seg_loss = criterion_seg(binary_seg_output, seg_labels)

        # Convert seg_labels to one-hot encoding for instance loss
        seg_labels_one_hot = one_hot_encoding(seg_labels, num_classes=2) # Assuming 2 classes: background and lane
        seg_labels_one_hot = seg_labels_one_hot.to(device) # Move one-hot labels to the GPU

        instance_loss = criterion_instance(embedding_output, seg_labels_one_hot)

        loss = seg_loss + instance_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_seg_loss += seg_loss.item()
        total_instance_loss += instance_loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\t"
                  f"Loss: {loss.item():.4f} (Seg: {seg_loss.item():.4f}, Instance: {instance_loss.item():.4f})")

    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_instance_loss = total_instance_loss / len(train_loader)
    return avg_loss, avg_seg_loss, avg_instance_loss

def validate(model, val_loader, criterion_seg, criterion_instance, device):
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_instance_loss = 0
    with torch.no_grad():
        for batch_idx, (images, seg_labels, lane_names) in enumerate(val_loader):
            # Move data to the GPU
            images, seg_labels = images.to(device), seg_labels.to(device)

            binary_seg_output, embedding_output = model(images)
            seg_loss = criterion_seg(binary_seg_output, seg_labels)

            # Convert seg_labels to one-hot encoding for instance loss
            seg_labels_one_hot = one_hot_encoding(seg_labels, num_classes=2) # Assuming 2 classes: background and lane
            seg_labels_one_hot = seg_labels_one_hot.to(device) # Move one-hot labels to the GPU

            instance_loss = criterion_instance(embedding_output, seg_labels_one_hot)

            loss = seg_loss + instance_loss

            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_instance_loss += instance_loss.item()
            print(f"Validating [{batch_idx}/{len(val_loader)}]\t"
                  f"Loss: {loss.item():.4f} (Seg: {seg_loss.item():.4f}, Instance: {instance_loss.item():.4f})")

    avg_loss = total_loss / len(val_loader)
    avg_seg_loss = total_seg_loss / len(val_loader)
    avg_instance_loss = total_instance_loss / len(val_loader)
    return avg_loss, avg_seg_loss, avg_instance_loss

def main():
    args = parse_args()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # Use "cuda:0" to specify the first GPU

    # Data transforms
    train_transforms = Compose([
        Resize((args.img_size, args.img_width)),
        RandomRotation(5, p=0.5),
        RandomHorizontalFlip(p=0.5),
        PhotometricDistort(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Example: ImageNet mean/std
        ToTensor()
    ])
    val_transforms = Compose([
        Resize((args.img_size, args.img_width)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use the same mean/std as training
        ToTensor()
    ])

    # Dataset
    train_dataset = CULaneDataset(args.data_dir, transforms=train_transforms)
    # For validation, you might want to load from other driver folders
    val_dataset = CULaneDataset(args.data_dir, transforms=val_transforms)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True) # Add pin_memory=True
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True) # Add pin_memory=True

    # Model
    model = LaneNet().to(device) # Move the model to the GPU

    # Load pre-trained ENet weights (if provided)
    if args.pretrained_weights:
        pretrained_dict = torch.load(args.pretrained_weights)
        model_dict = model.encoder.state_dict()
        # Filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # Overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # Load the new state dict
        model.encoder.load_state_dict(model_dict)

    # Loss functions
    criterion_seg = BinaryFocalLoss().to(device)  # Move loss functions to the GPU
    criterion_instance = DiscriminativeLoss().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print('Training')
        start_time = time.time()
        train_loss, train_seg_loss, train_instance_loss = train(model, train_loader, criterion_seg, criterion_instance, optimizer, epoch, device)
        print('Validating')
        val_loss, val_seg_loss, val_instance_loss = validate(model, val_loader, criterion_seg, criterion_instance, device)
        end_time = time.time()

        print(f"Epoch: {epoch}/{args.epochs}  "
              f"Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Instance: {train_instance_loss:.4f})  "
              f"Val Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Instance: {val_instance_loss:.4f})  "
              f"Time: {end_time - start_time:.2f}s")

        print('Saving checkpoint')
        # Save checkpoint
        if epoch % 1 == 0:  # Save every epoch, for example
            checkpoint_path = os.path.join(f"{CHECKPOINTS_DIR}/{VERSION}_epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    main()