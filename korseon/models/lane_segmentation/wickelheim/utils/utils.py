import torch
import numpy as np

def one_hot_encoding(label, num_classes):
    """
    Converts a segmentation label (N, H, W) to a one-hot encoded tensor (N, C, H, W).

    Args:
        label (torch.Tensor): Segmentation label tensor of shape (N, H, W).
        num_classes (int): Number of classes.

    Returns:
        torch.Tensor: One-hot encoded tensor of shape (N, C, H, W).
    """
    # Create an empty one-hot tensor
    one_hot = torch.zeros(label.size(0), num_classes, label.size(1), label.size(2), dtype=torch.float32, device=label.device)

    # Scatter ones for each class
    one_hot.scatter_(1, label.unsqueeze(1).long(), 1.0)

    return one_hot