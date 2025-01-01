import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from model.lanenet import LaneNet
from model.enet import ENet
from dataset.culane import CULaneDataset  # Or your dataset loader
from utils.loss import BinaryFocalLoss, DiscriminativeLoss
from utils.transforms import *
from utils.utils import one_hot_encoding
import time
from torch.cuda.amp import GradScaler, autocast  # For mixed-precision training
from pytorchtools import EarlyStopping  # Assuming you saved the EarlyStopping class in pytorchtools.py

# Constants and Configuration
GEN = 'Wickelheim'
VERSION = 'W3'
EPOCHS = 50
CHECKPOINTS_DIR = f'./drive/MyDrive/ColabNotebooks/{GEN}/{VERSION}/checkpoints'
EARLY_STOPPING_PATIENCE = 5  # Number of epochs to wait for improvement before stopping

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="culane", help="Dataset to use (culane, bdd100k, etc.)")
    parser.add_argument("--data_dir", type=str, default="./data/culane", help="Path to dataset directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")  # Increased batch size
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Learning rate")  # Reduced learning rate
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loader workers")
    parser.add_argument("--save_dir", type=str, default=CHECKPOINTS_DIR, help="Directory to save checkpoints")
    parser.add_argument("--img_size", type=int, default=256, help="height of resized image")
    parser.add_argument("--img_width", type=int, default=448, help="width of resized image")
    parser.add_argument("--pretrained_weights", type=str, help="Path to pretrained ENet weights (optional)")
    parser.add_argument("--use_amp", action="store_true", help="Enable Automatic Mixed Precision (AMP)")  # Add flag for AMP
    return parser.parse_args()

def train(model, train_loader, criterion_seg, criterion_instance, optimizer, epoch, device, use_amp, scaler):
    model.train()
    total_loss = 0
    total_seg_loss = 0
    total_instance_loss = 0
    accumulation_steps = 4

    for batch_idx, (images, seg_labels, lane_names) in enumerate(train_loader):
        images, seg_labels = images.to(device), seg_labels.to(device)

        # Use autocast for mixed-precision (conditional)
        with autocast(enabled=use_amp):
            binary_seg_output, embedding_output = model(images)

            # Print logits before sigmoid
            logits = binary_seg_output.detach()
            print("Logits (min, max, mean):", logits.min(), logits.max(), logits.mean())

            seg_loss = criterion_seg(binary_seg_output, seg_labels)

            seg_labels_one_hot = one_hot_encoding(seg_labels, num_classes=2)
            seg_labels_one_hot = seg_labels_one_hot.to(device)

            instance_loss = criterion_instance(embedding_output, seg_labels_one_hot)

            loss = seg_loss + instance_loss
            loss = loss / accumulation_steps

        # Print values for debugging
        if batch_idx % 10 == 0:
            print("binary_seg_output (min, max, mean):", binary_seg_output.min(), binary_seg_output.max(), binary_seg_output.mean())
            print("seg_loss:", seg_loss)
            print("instance_loss:", instance_loss)
            print("total loss:", loss)

        if use_amp:
            # Scale the loss and perform backward pass
            scaler.scale(loss).backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (batch_idx + 1) % accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
        else:
            # Perform backward pass (without AMP)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        total_seg_loss += seg_loss.item()
        total_instance_loss += instance_loss.item()

        if batch_idx % 10 == 0:
            print(f"Epoch: {epoch} [{batch_idx}/{len(train_loader)}]\t"
                  f"Loss: {loss.item() * accumulation_steps:.4f} (Seg: {seg_loss.item():.4f}, Instance: {instance_loss.item():.4f})")

    # Make sure to do an optimizer step for the last batch
    if len(train_loader) % accumulation_steps != 0:
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(train_loader)
    avg_seg_loss = total_seg_loss / len(train_loader)
    avg_instance_loss = total_instance_loss / len(train_loader)
    return avg_loss, avg_seg_loss, avg_instance_loss

def validate(model, val_loader, criterion_seg, criterion_instance, device, use_amp):
    model.eval()
    total_loss = 0
    total_seg_loss = 0
    total_instance_loss = 0
    with torch.no_grad():
        for batch_idx, (images, seg_labels, lane_names) in enumerate(val_loader):
            images, seg_labels = images.to(device), seg_labels.to(device)

            # Use autocast for mixed-precision (conditional)
            with autocast(enabled=use_amp):
                binary_seg_output, embedding_output = model(images)

                # Print logits before sigmoid
                logits = binary_seg_output.detach()
                print("Logits (min, max, mean):", logits.min(), logits.max(), logits.mean())

                seg_loss = criterion_seg(binary_seg_output, seg_labels)

                seg_labels_one_hot = one_hot_encoding(seg_labels, num_classes=2)
                seg_labels_one_hot = seg_labels_one_hot.to(device)

                instance_loss = criterion_instance(embedding_output, seg_labels_one_hot)

                loss = seg_loss + instance_loss

            total_loss += loss.item()
            total_seg_loss += seg_loss.item()
            total_instance_loss += instance_loss.item()

            # Print values for debugging
            if batch_idx % 10 == 0:
                print("binary_seg_output (min, max, mean):", binary_seg_output.min(), binary_seg_output.max(), binary_seg_output.mean())
                print("seg_loss:", seg_loss)
                print("instance_loss:", instance_loss)
                print("total loss:", loss)

            print(f"Validating [{batch_idx}/{len(val_loader)}]\t"
                  f"Loss: {loss.item():.4f} (Seg: {seg_loss.item():.4f}, Instance: {instance_loss.item():.4f})")

    avg_loss = total_loss / len(val_loader)
    avg_seg_loss = total_seg_loss / len(val_loader)
    avg_instance_loss = total_instance_loss / len(val_loader)
    return avg_loss, avg_seg_loss, avg_instance_loss

def main():
    args = parse_args()

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Data transforms
    train_transforms = Compose([
        Resize((args.img_size, args.img_width)),
        RandomRotation(5, p=0.5),
        RandomHorizontalFlip(p=0.5),
        PhotometricDistort(p=0.5),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])
    val_transforms = Compose([
        Resize((args.img_size, args.img_width)),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensor()
    ])

    # Dataset
    train_dataset = CULaneDataset(args.data_dir, image_set='train', transforms=train_transforms)
    val_dataset = CULaneDataset(args.data_dir, image_set='val', transforms=val_transforms)

    # Data Loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # Model
    model = LaneNet().to(device)

    # Load pre-trained ENet weights (if provided)
    if args.pretrained_weights:
        pretrained_dict = torch.load(args.pretrained_weights)
        model_dict = model.encoder.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.encoder.load_state_dict(model_dict)

    # Loss functions
    criterion_seg = BinaryFocalLoss().to(device)
    criterion_instance = DiscriminativeLoss().to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-4) # Add weight decay

    # Learning Rate Scheduler
    lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, verbose=True)

    # Create a GradScaler for mixed-precision (conditional)
    scaler = GradScaler() if args.use_amp else None

    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=EARLY_STOPPING_PATIENCE, verbose=True, path=os.path.join(args.save_dir, f'early_stopping_checkpoint_{VERSION}.pth'))

    # Create checkpoint directory if it doesn't exist
    os.makedirs(args.save_dir, exist_ok=True)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        print('Training')
        start_time = time.time()
        train_loss, train_seg_loss, train_instance_loss = train(model, train_loader, criterion_seg, criterion_instance, optimizer, epoch, device, args.use_amp, scaler)
        print('Validating')
        val_loss, val_seg_loss, val_instance_loss = validate(model, val_loader, criterion_seg, criterion_instance, device, args.use_amp)
        end_time = time.time()

        # Update learning rate scheduler based on validation loss
        lr_scheduler.step(val_loss)

        print(f"Epoch: {epoch}/{args.epochs}  "
              f"Train Loss: {train_loss:.4f} (Seg: {train_seg_loss:.4f}, Instance: {train_instance_loss:.4f})  "
              f"Val Loss: {val_loss:.4f} (Seg: {val_seg_loss:.4f}, Instance: {val_instance_loss:.4f})  "
              f"Time: {end_time - start_time:.2f}s")

        # Check for early stopping
        early_stopping(val_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        print('Saving checkpoint')
        # Save checkpoint
        checkpoint_path = os.path.join(args.save_dir, f"{VERSION}_epoch_{epoch}.pth")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True) # Enable anomaly detection for debugging
    main()