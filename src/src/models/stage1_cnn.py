import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)

class Stage1CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64, stride=2)
        self.enc3 = ConvBlock(64, 128, stride=2)
        self.enc4 = ConvBlock(128, 256, stride=2)

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(256, 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)

        x = self.pool(x).flatten(1)
        x = self.fc(x)

        return x  # logits
