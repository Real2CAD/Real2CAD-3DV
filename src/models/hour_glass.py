from typing import Tuple

import torch
import torch.nn as nn


# encoder - decoder structure
class HourGlass(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(x)

        out = self.decoder(hidden)
        return out

# with skip connection
class HourGlassSkip(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feature_maps = self.encoder(x)
        hidden = feature_maps[min(feature_maps.keys())] # last layer of encoder

        out = self.decoder(hidden, feature_maps)
        return out


# encoder - decoder structure, also output bottleneck hidden layer for operations such as classification
class HourGlassMultiOut(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.encoder(x)

        # feed hidden layer to a classification conv

        out = self.decoder(hidden)
        return out, hidden

# with skip connection
class HourGlassMultiOutSkip(nn.Module):
    def __init__(self, encoder: nn.Module, decoder: nn.Module) -> None:
        super().__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feature_maps = self.encoder(x)
        hidden = feature_maps[min(feature_maps.keys())] # last layer of encoder

        # feed hidden layer to a classification conv

        out = self.decoder(hidden, feature_maps)
        return out, hidden