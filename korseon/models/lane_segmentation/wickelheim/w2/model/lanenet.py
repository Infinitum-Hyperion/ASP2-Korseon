import torch
import torch.nn as nn
import torch.nn.functional as F
from .enet import ENet

class LaneNet(nn.Module):
    def __init__(self, num_classes=2, encoder_relu=False, decoder_relu=True):
        super(LaneNet, self).__init__()
        self.num_classes = num_classes
        self.encoder = ENet(num_classes=num_classes)
        self.encoder_relu = encoder_relu
        self.decoder_relu = decoder_relu

        self.binary_seg = nn.Conv2d(num_classes, 1, kernel_size=1)

        # Instance segmentation branch
        self.embedding = nn.Sequential(
            nn.Conv2d(num_classes, 4, kernel_size=1),
            nn.InstanceNorm2d(4)
        )

    def forward(self, input):
        # Encoder
        output = self.encoder(input)

        # Binary segmentation
        binary_seg_output = self.binary_seg(output)
        if self.decoder_relu:
            binary_seg_output = F.relu(binary_seg_output)

        # Instance segmentation
        embedding_output = self.embedding(output)

        return binary_seg_output, embedding_output