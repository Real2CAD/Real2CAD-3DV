from typing import List

import torch
import torch.nn as nn

from models import ResNetBlock


class ResNetDecoder(nn.Module):
    def __init__(self, num_output_channels: int, num_features: List = None, verbose: bool = False) -> None:
        super().__init__()

        if num_features is None:
            num_features = [8, 16, 32, 64, 256]

        self.verbose = verbose
        self.num_features = [num_output_channels] + num_features

        self.network = nn.Sequential(
            
            # [layer 0] shape: 1 -> 2, feature: 256 -> 64
            nn.ReLU(),
            nn.ConvTranspose3d(self.num_features[5], self.num_features[4], kernel_size=2, stride=1, padding=0,
                               bias=False),

            # [layer 1] shape: 2 -> 4, feature: 64 -> 32
            # ResNetBlock(self.num_features[4]),
            ResNetBlock(self.num_features[4]),
            nn.ConvTranspose3d(self.num_features[4], self.num_features[3], kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[3]),

            # [layer 2] shape: 4 -> 8, feature: 32 -> 16
            # ResNetBlock(self.num_features[3]),
            ResNetBlock(self.num_features[3]),
            nn.ConvTranspose3d(self.num_features[3], self.num_features[2], kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[2]),

            # [layer 3] shape: 8 -> 16, feature: 16 -> 8 
            # ResNetBlock(self.num_features[2]),
            ResNetBlock(self.num_features[2]),
            nn.ConvTranspose3d(self.num_features[2], self.num_features[1], kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),

            # [layer 4] shape: 16 -> 32, feature: 8 -> 8 
            # ResNetBlock(self.num_features[1]),
            ResNetBlock(self.num_features[1]),
            nn.ConvTranspose3d(self.num_features[1], self.num_features[1], kernel_size=4, stride=2, padding=1,
                               bias=False),

            # [layer 5] shape: 32 -> 32, feature: 8 -> 1 
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]),
            nn.ConvTranspose3d(self.num_features[1], self.num_features[0], kernel_size=7, stride=1, padding=3,
                               bias=False),
        )
        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
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


class ResNetDecoderSkip(nn.Module):
    def __init__(self, num_output_channels: int, num_features: List = None, skip_layer: List = None, verbose: bool = True) -> None:
        super().__init__()

        if num_features is None:
            num_features = [8, 16, 32, 64, 256]
        
        if skip_layer is None: 
            skip_layer = [False, True, True, False, False]
            #skip_layer = [False, True, True, True, False]

        self.verbose = verbose
        self.num_features = [num_output_channels] + num_features
        self.skip_layer = skip_layer

        # shape: 1 -> 2, feature: 256 -> 64
        self.layer0 = nn.Sequential(
            nn.ReLU(),
            nn.ConvTranspose3d(self.num_features[5], self.num_features[4], kernel_size=2, stride=1, padding=0,
                               bias=False))
        
        # shape: 2 -> 4, feature: 64 -> 32
        if self.skip_layer[0]:
            in_ch = self.num_features[4] * 2
        else:
            in_ch = self.num_features[4]
        self.layer1 = nn.Sequential(
            ResNetBlock(in_ch),
            nn.ConvTranspose3d(in_ch, self.num_features[3], kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[3]))
        
        # shape: 4 -> 8, feature: 32 + 32 -> 16 (skip connection)
        if self.skip_layer[1]:
            in_ch = self.num_features[3] * 2
        else:
            in_ch = self.num_features[3]
        self.layer2 = nn.Sequential(
            ResNetBlock(in_ch),
            nn.ConvTranspose3d(in_ch, self.num_features[2], kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[2]))

        # shape: 8 -> 16, feature: 16 + 16 -> 8 (skip connection)
        if self.skip_layer[2]:
            in_ch = self.num_features[2] * 2
        else:
            in_ch = self.num_features[2]
        self.layer3 = nn.Sequential(
            ResNetBlock(in_ch),
            nn.ConvTranspose3d(in_ch, self.num_features[1], kernel_size=4, stride=2, padding=1,
                               bias=False),
            nn.ReLU(),
            nn.BatchNorm3d(self.num_features[1]))

        # shape: 16 -> 32, feature: 8 + 8 -> 8 (skip connection)
        if self.skip_layer[3]:
            in_ch = self.num_features[1] * 2
        else:
            in_ch = self.num_features[1]
        self.layer4 = nn.Sequential(
            ResNetBlock(in_ch),
            nn.ConvTranspose3d(in_ch, self.num_features[1], kernel_size=4, stride=2, padding=1,
                               bias=False))
        
        # shape: 32 -> 32, feature: 8 -> 1 
        if self.skip_layer[4]:
            in_ch = self.num_features[1] * 2
        else:
            in_ch = self.num_features[1]
        self.layer5 = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm3d(in_ch),
            nn.ConvTranspose3d(in_ch, self.num_features[0], kernel_size=7, stride=1, padding=3,
                               bias=False)
        )

        self.init_weights()

    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        
        x = self.layer0(x)
        
        if self.skip_layer[0]:
            x = torch.cat((x, skip[2]), dim=1) 

        x = self.layer1(x)
        
        if self.skip_layer[1]:
            x = torch.cat((x, skip[4]), dim=1) 

        x = self.layer2(x)
        
        if self.skip_layer[2]:
            x = torch.cat((x, skip[8]), dim=1) 

        x = self.layer3(x)
        
        if self.skip_layer[3]:
            x = torch.cat((x, skip[16]), dim=1) 

        x = self.layer4(x)

        if self.skip_layer[4]:
            x = torch.cat((x, skip[32]), dim=1)

        x = self.layer5(x)

        return x