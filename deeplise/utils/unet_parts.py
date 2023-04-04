""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
from echoAI.Activation.Torch.mish import Mish
from torch.nn.utils import weight_norm

class ConvBlock(nn.Module):
    """(convolution => [BN] => ELU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        scaleFactor = out_channels // 4

        self.conv1 = nn.Conv3d(in_channels, scaleFactor, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(scaleFactor + in_channels, scaleFactor, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d((scaleFactor * 2) + in_channels, scaleFactor, kernel_size=3, padding=1)
        self.conv4 = nn.Conv3d((scaleFactor * 3) + in_channels, scaleFactor, kernel_size=3, padding=1)

        self.mish = Mish()

    def forward(self, x):

        # x1 = F.elu(self.conv1(x))
        # x2 = F.elu(self.conv2(torch.cat([x, x1], dim=1)))
        # x3 = F.elu(self.conv3(torch.cat([x, x1, x2], dim=1)))
        # x4 = F.elu(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))

        x1 = self.mish(self.conv1(x))
        x2 = self.mish(self.conv2(torch.cat([x, x1], dim=1)))
        x3 = self.mish(self.conv3(torch.cat([x, x1, x2], dim=1)))
        x4 = self.mish(self.conv4(torch.cat([x, x1, x2, x3], dim=1)))

        return torch.cat([x1, x2, x3, x4], dim=1)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            ConvBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose3d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv1 = ConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW

        diffZ = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffY = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffX = torch.tensor([x2.size()[4] - x1.size()[4]])

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2,
                        diffZ // 2, diffZ - diffZ // 2])
                        
        x = torch.cat([x2, x1], dim=1)
        x = self.conv1(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
