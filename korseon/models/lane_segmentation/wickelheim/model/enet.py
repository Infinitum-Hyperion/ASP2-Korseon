import torch
import torch.nn as nn

class InitialBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InitialBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - 3, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x):
        x_conv = self.conv(x)
        x_pool = self.pool(x)
        x = torch.cat([x_conv, x_pool], dim=1)
        x = self.bn(x)
        x = self.act(x)
        return x

class RegularBottleneck(nn.Module):
    def __init__(self, channels, internal_ratio=4, kernel_size=3, padding=1, dilation=1, asymmetric=False, dropout_prob=0.1):
        super(RegularBottleneck, self).__init__()
        internal_channels = channels // internal_ratio
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(channels, internal_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels)
        )
        if asymmetric:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(kernel_size, 1), stride=1, padding=(padding, 0), dilation=dilation, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, kernel_size), stride=1, padding=(0, padding), dilation=dilation, bias=False),
            )
        else:
            self.ext_conv2 = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, dilation=dilation, bias=False)
            )
        self.ext_conv3 = nn.Sequential(
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels),
            nn.Conv2d(internal_channels, channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.ext_regul = nn.PReLU(channels)

    def forward(self, x):
        x_main = self.ext_conv1(x)
        x_main = self.ext_conv2(x_main)
        x_main = self.ext_conv3(x_main)
        x = x + x_main
        x = self.ext_regul(x)
        return x

class DownsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3, padding=1, return_indices=False, dropout_prob=0.1):
        super(DownsamplingBottleneck, self).__init__()
        internal_channels = in_channels // internal_ratio
        self.return_indices = return_indices
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=kernel_size, stride=2, padding=padding, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels)
        )
        self.ext_conv2 = nn.Sequential(
            nn.Conv2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels)
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.ext_pool = nn.MaxPool2d(2, stride=2, return_indices=return_indices)
        self.ext_regul = nn.PReLU(out_channels)

        # 1x1 Convolution for x_skip to match dimensions
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2, bias=False)
        self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x_main = self.ext_conv1(x)
        x_main = self.ext_conv2(x_main)
        x_main = self.ext_conv3(x_main)

        x_skip = self.conv1x1(x)
        x_skip = self.bn_skip(x_skip)

        x = x_main + x_skip
        x = self.ext_regul(x)
        if self.return_indices:
            return x, x_main.indices
        else:
            return x

class UpsamplingBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, internal_ratio=4, kernel_size=3, padding=1, dropout_prob=0.1):
        super(UpsamplingBottleneck, self).__init__()
        internal_channels = in_channels // internal_ratio
        self.ext_conv1 = nn.Sequential(
            nn.Conv2d(in_channels, internal_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels)
        )
        self.ext_conv2 = nn.Sequential(
            nn.ConvTranspose2d(internal_channels, internal_channels, kernel_size=kernel_size, stride=2, padding=padding, output_padding=1, bias=False),
            nn.BatchNorm2d(internal_channels),
            nn.PReLU(internal_channels)
        )
        self.ext_conv3 = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=dropout_prob)
        )
        self.ext_regul = nn.PReLU(out_channels)

        # 1x1 Convolution for x_skip to match dimensions
        self.conv1x1 = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, stride=2, output_padding=1, bias=False)
        self.bn_skip = nn.BatchNorm2d(out_channels)

    def forward(self, x, indices):
        x_main = self.ext_conv1(x)
        x_main = self.ext_conv2(x_main)
        x_main = self.ext_conv3(x_main)
        x_skip = self.conv1x1(x)
        x_skip = self.bn_skip(x_skip)
        x = x_main + x_skip
        x = self.ext_regul(x)
        return x

class ENet(nn.Module):
    def __init__(self, num_classes):
        super(ENet, self).__init__()
        self.initial_block = InitialBlock(3, 16)
        self.downsample1_0 = DownsamplingBottleneck(16, 64, return_indices=True, dropout_prob=0.01)
        self.regular1_1 = RegularBottleneck(64, dropout_prob=0.01)
        self.regular1_2 = RegularBottleneck(64, dropout_prob=0.01)
        self.regular1_3 = RegularBottleneck(64, dropout_prob=0.01)
        self.regular1_4 = RegularBottleneck(64, dropout_prob=0.01)
        self.downsample2_0 = DownsamplingBottleneck(64, 128, return_indices=True, dropout_prob=0.1)
        self.regular2_1 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated2_2 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric2_3 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated2_4 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular2_5 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated2_6 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric2_7 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1)
        self.dilated2_8 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)
        self.regular3_0 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated3_1 = RegularBottleneck(128, dilation=2, padding=2, dropout_prob=0.1)
        self.asymmetric3_2 = RegularBottleneck(128, kernel_size=5, padding=2, asymmetric=True, dropout_prob=0.1)
        self.dilated3_3 = RegularBottleneck(128, dilation=4, padding=4, dropout_prob=0.1)
        self.regular3_4 = RegularBottleneck(128, dropout_prob=0.1)
        self.dilated3_5 = RegularBottleneck(128, dilation=8, padding=8, dropout_prob=0.1)
        self.asymmetric3_6 = RegularBottleneck(128, kernel_size=5, asymmetric=True, padding=2, dropout_prob=0.1)
        self.dilated3_7 = RegularBottleneck(128, dilation=16, padding=16, dropout_prob=0.1)
        self.upsample4_0 = UpsamplingBottleneck(128, 64, dropout_prob=0.1)
        self.regular4_1 = RegularBottleneck(64, dropout_prob=0.1)
        self.regular4_2 = RegularBottleneck(64, dropout_prob=0.1)
        self.upsample5_0 = UpsamplingBottleneck(64, 16, dropout_prob=0.1)
        self.regular5_1 = RegularBottleneck(16, dropout_prob=0.1)
        self.transposed_conv = nn.ConvTranspose2d(16, num_classes, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)

    def forward(self, x):
        x = self.initial_block(x)
        x, indices1 = self.downsample1_0(x)
        x = self.regular1_1(x)
        x = self.regular1_2(x)
        x = self.regular1_3(x)
        x = self.regular1_4(x)
        x, indices2 = self.downsample2_0(x)
        x = self.regular2_1(x)
        x = self.dilated2_2(x)
        x = self.asymmetric2_3(x)
        x = self.dilated2_4(x)
        x = self.regular2_5(x)
        x = self.dilated2_6(x)
        x = self.asymmetric2_7(x)
        x = self.dilated2_8(x)
        x = self.regular3_0(x)
        x = self.dilated3_1(x)
        x = self.asymmetric3_2(x)
        x = self.dilated3_3(x)
        x = self.regular3_4(x)
        x = self.dilated3_5(x)
        x = self.asymmetric3_6(x)
        x = self.dilated3_7(x)
        x = self.upsample4_0(x, indices2)
        x = self.regular4_1(x)
        x = self.regular4_2(x)
        x = self.upsample5_0(x, indices1)
        x = self.regular5_1(x)
        x = self.transposed_conv(x)
        return x