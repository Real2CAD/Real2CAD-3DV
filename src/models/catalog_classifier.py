import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from operator import itemgetter 

# reference:
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# https://stackoverflow.com/questions/62456558/is-one-hot-encoding-required-for-using-pytorchs-cross-entropy-loss-function

# two fc layers (with dropout)
class CatalogClassifier(nn.Module):

    def __init__(self, input_size: List[int], num_classes: int) -> None:
        super(CatalogClassifier, self).__init__()
        self.selected_categories: List[str] = ["03001627", "04379243", "02747177", "02818832", "02871439", "02933112", "04256520", "other"]
                                   
        self.input_size = input_size
        self.fc1 = nn.Linear(torch.prod(torch.tensor(input_size)).item(), num_classes * 10)
        self.fc2 = nn.Linear(num_classes * 10, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict_idx(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward(x)
        _, y = torch.max(x, 1)
        return y # [0 - 7]

    def predict_name(self, x: torch.Tensor) -> str:
        y = self.predict_idx(x) # [0 - 7]
        #predicted = self.selected_categories[y] # to class string
        predicted = itemgetter(*y)(self.selected_categories) # to class string
        return predicted


class MultiHeadModule(nn.Module):
    num_sym_category: int = 4

    def __init__(self, input_size: List[int], num_category: int, train_symmetry: bool=False, train_scale: bool=False):
        super(MultiHeadModule, self).__init__()
        self.selected_categories: List[str] = ["03001627", "04379243", "02747177", "02818832", "02871439", "02933112",
                                               "04256520", "other"]
        self.train_symmetry = train_symmetry
        self.train_scale = train_scale

        self.input_size = input_size
        self.shared_fc1 = nn.Linear(torch.prod(torch.tensor(input_size)).item(), 80)
        self.dropout = nn.Dropout(p=0.2)
        self.cat_fc2 = nn.Linear(80, num_category)
        self.dropout2 = nn.Dropout(p=0.2)
        self.sym_fc2 = nn.Linear(80, self.num_sym_category)
        self.dropout3 = nn.Dropout(p=0.2)
        self.scale_fc2 = nn.Linear(80, 10)
        self.scale_fc3 = nn.Linear(10, 3)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, 1)
        x = F.relu(self.shared_fc1(x))

        # category classifier
        cat = self.dropout(x)
        cat = self.cat_fc2(cat)
        result_dict = {'category': cat}

        # symmetry classifier
        if self.train_symmetry:
            sym = self.dropout2(x)
            sym = self.sym_fc2(sym)
            result_dict['symmetry'] = sym

        # scale regressor
        if self.train_scale:
            scale = self.dropout3(x)
            scale = F.relu(self.scale_fc2(scale))
            scale = self.scale_fc3(scale)
            result_dict['scale'] = scale
        return result_dict

    def predict_idx(self, x: torch.Tensor) -> torch.Tensor:
        result_dict = self.forward(x)
        cat = result_dict['category']
        _, y = torch.max(cat, 1)
        return y # [0 - 7]

    def predict_name(self, x: torch.Tensor) -> str:
        y = self.predict_idx(x) # [0 - 7]
        #predicted = self.selected_categories[y] # to class string
        predicted = itemgetter(*y)(self.selected_categories) # to class string
        return predicted
