from typing import List

import torch
import torch.nn as nn

from models import ResNetBlock


class ResNetEncoder(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_features: List = None, verbose: bool = False) -> None:
        super().__init__()

        if num_features is None:
            num_features = [8, 16, 32, 64, 256]

        self.verbose = verbose
        self.num_features = [num_input_channels] + num_features

        # Relu -> batchnorm -> conv3d -> resblock

        self.network = nn.Sequential(
            # [layer 0] shape: 32 -> 32, feature: 1 -> 8
            nn.Conv3d(self.num_features[0], self.num_features[1], kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),

            # [layer 1] shape: 32 -> 16, feature: 8 -> 8
            nn.Conv3d(self.num_features[1], self.num_features[1], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[1]),

            # [layer 2] shape: 16 -> 8, feature: 8 -> 16
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),
            nn.Conv3d(self.num_features[1], self.num_features[2], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[2]),

            # [layer 3] shape: 8 -> 4, feature: 16 -> 32
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[2]),
            nn.Conv3d(self.num_features[2], self.num_features[3], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[3]),

            # [layer 4] shape: 4 -> 2, feature: 32 -> 64
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[3]),
            nn.Conv3d(self.num_features[3], self.num_features[4], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[4]),

            # [layer 5] shape: 2 -> 1, feature: 64 -> 256
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[4]),
            nn.Conv3d(self.num_features[4], self.num_features[5], kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.num_features[5])
        )
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers = list(self.network.children())
        for depth, layer in enumerate(layers):
            shape_before = x.data[0].size()
            x = layer(x)
            shape_after = x.data[0].size()

            if self.verbose is True:
                print(f"Layer {depth}: {shape_before} --> {shape_after}")
                # self.verbose = False

        return x


class ResNetEncoderSkip(nn.Module):
    def __init__(self, num_input_channels: int = 1, num_features: List = None, verbose: bool = True) -> None:
        super().__init__()

        if num_features is None:
            num_features = [8, 16, 32, 64, 256]

        self.verbose = verbose
        self.num_features = [num_input_channels] + num_features

        # Relu -> batchnorm -> conv3d -> resblock
        
        # shape: 32 -> 32, feature: 1 -> 8
        self.layer0 = nn.Sequential(
            nn.Conv3d(self.num_features[0], self.num_features[1], kernel_size=7, stride=1, padding=3, bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]))
        
        # shape: 32 -> 16, feature: 8 -> 8
        self.layer1 = nn.Sequential(
            nn.Conv3d(self.num_features[1], self.num_features[1], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[1]))
        
        # shape: 16 -> 8, feature: 8 -> 16
        self.layer2 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),
            nn.Conv3d(self.num_features[1], self.num_features[2], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[2]))
        
        # shape: 8 -> 4, feature: 16 -> 32
        self.layer3 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[2]),
            nn.Conv3d(self.num_features[2], self.num_features[3], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[3])
        )
        
        # shape: 4 -> 2, feature: 32 -> 64
        self.layer4 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[3]),
            nn.Conv3d(self.num_features[3], self.num_features[4], kernel_size=4, stride=2, padding=1, bias=False),
            ResNetBlock(self.num_features[4]))
        
        # shape: 2 -> 1, feature: 64 -> 256
        self.layer5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[4]),
            nn.Conv3d(self.num_features[4], self.num_features[5], kernel_size=2, stride=1, padding=0, bias=False),
            nn.BatchNorm3d(self.num_features[5]))

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        out = {} # record each layer's feature map for skip connection
        
        x = self.layer0(x)
        out[x.data[0].size()[3]] = x

        x = self.layer1(x)
        out[x.data[0].size()[3]] = x

        x = self.layer2(x)
        out[x.data[0].size()[3]] = x

        x = self.layer3(x)
        out[x.data[0].size()[3]] = x

        x = self.layer4(x)
        out[x.data[0].size()[3]] = x

        x = self.layer5(x)
        out[x.data[0].size()[3]] = x

        return out