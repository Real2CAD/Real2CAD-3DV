from typing import Tuple

import torch
import torch.nn as nn


class TripletNet(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor = self.network(anchor)
        positive = self.network(positive)
        negative = self.network(negative)

        return anchor, positive, negative

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)



class TripletNetBatch(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) ->  torch.Tensor:
        anchor = self.network(anchor)
        positive = self.network(positive)
    
        return torch.cat((anchor, positive), dim=0) 

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)


class TripletNetBatchMix(nn.Module):
    def __init__(self, network: nn.Module) -> None:
        super().__init__()
        self.network = network

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor) ->  torch.Tensor:
        
        half_shape = int(anchor.shape[0] / 2)
        anchor = self.network(anchor)[0:half_shape-1,-1]
        positive = self.network(positive)[0:half_shape-1,-1]
    
        return torch.cat((anchor, positive), dim=0) 

    def embed(self, data: torch.Tensor) -> torch.Tensor:
        return self.network(data)