# Part 3: Heatmap-based Keypoint Detection

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two consecutive conv-bn-relu blocks."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, num_keypoints=68, heatmap_size=64):
        super().__init__()
        self.heatmap_size = heatmap_size

        # Encoder
        self.enc1 = DoubleConv(in_channels, 64)
        self.enc2 = DoubleConv(64, 128)
        self.enc3 = DoubleConv(128, 256)
        self.enc4 = DoubleConv(256, 512)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = DoubleConv(512, 1024)

        # Decoder
        self.up4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = DoubleConv(1024, 512)
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = DoubleConv(512, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)

        # Output: one heatmap per keypoint
        self.out_conv = nn.Conv2d(64, num_keypoints, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)                     # [B, 64, 224, 224]
        e2 = self.enc2(self.pool(e1))          # [B, 128, 112, 112]
        e3 = self.enc3(self.pool(e2))          # [B, 256, 56, 56]
        e4 = self.enc4(self.pool(e3))          # [B, 512, 28, 28]

        # Bottleneck
        b = self.bottleneck(self.pool(e4))     # [B, 1024, 14, 14]

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))   # [B, 512, 28, 28]
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))  # [B, 256, 56, 56]
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))  # [B, 128, 112, 112]
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))  # [B, 64, 224, 224]

        # Output heatmaps resized to target resolution
        out = self.out_conv(d1)                # [B, 68, 224, 224]
        out = F.interpolate(out, size=self.heatmap_size, mode='bilinear', align_corners=False)
        return out  # [B, 68, 64, 64]
