from typing import List
import os
import getpass
from torch.optim.optimizer import Optimizer
import torch
import torch.nn as nn
import numpy as np

# set up weight and bias
def setup_wandb():
    username = getpass.getuser()
    print(username)
    wandb_key_path = "configs/" + username + "_wandb.key"
    if not os.path.exists(wandb_key_path):
        wandb_key = input(
            "[You need to firstly setup and login wandb] Please enter your wandb key (https://wandb.ai/authorize):")
        with open(wandb_key_path, "w") as fh:
            fh.write(wandb_key)
    else:
        print("wandb key already set")
    os.system("export WANDB_API_KEY=$(cat \"" + wandb_key_path + "\")")

def stepwise_learning_rate_decay(optimizer: Optimizer, learning_rate: float, iteration_number: int,
                                 steps: List, reduce: float = 0.1) -> float:
    if iteration_number in steps:
        steps.remove(iteration_number)
        learning_rate *= reduce
        print("Reduce learning rate to {}".format(learning_rate))

        for param in optimizer.param_groups:
            param["lr"] = learning_rate

    return learning_rate

def num_model_weights(model: nn.Module) -> int:
    num_weights = int(sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, model.parameters())]))
    return num_weights

def freeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = False

def unfreeze_model(model: nn.Module):
    for child in model.children():
        for param in child.parameters():
            param.requires_grad = True

def batch_save_checkpoint(separation_model, completion_model, classification_model, metric_model, optimizer, run_path, checkpoint_name, iter, epoch):
    torch.save({
            'epoch': epoch + 1,
            'iteration': iter + 1,
            'separation': separation_model.state_dict(),
            'completion': completion_model.state_dict(),
            'classification': classification_model.state_dict(),
            'metriclearning': metric_model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, os.path.join(run_path, f"{checkpoint_name}.pt"))


def get_category_idx(cat_str):
    category_idx_dict_cls = {"03001627": 0, "04379243": 1, "02747177": 2, "02818832": 3, "02871439": 4, "02933112": 5, "04256520": 6}
    # classification idx

    return category_idx_dict_cls.get(cat_str, 7) # "other": 7

def wandb_color_lut(cat_str):
    category_idx_dict_label = {"03001627": 1, "04379243": 2, "02747177": 8, "02818832": 4, "02871439": 5, "02933112": 6, "04256520": 7}
    # label in wandb [unfortunately, due to some bug, label 2 and 3 are both green in wandb]

    return category_idx_dict_label.get(cat_str, 9) # "other": 9

def get_category_name(cat_str):
    category_name_dict = {"03001627": "Chair", "04379243": "Table", "02747177": "Trash bin", "02818832": "Bed",
                      "02871439": "Bookshelf", "02933112": "Cabinet", "04256520": "Sofa"}

    return category_name_dict.get(cat_str, "Other")

def get_category_code_from_2d3ds(cat_name):
    #["chair", "table", "bookcase", "sofa"]
    category_code_dict = {"chair": "03001627", "table": "04379243", 
                      "bookcase": "02871439", "sofa": "04256520"}

    return category_code_dict.get(cat_name, "Other")


def get_category_name_verbose(cat_str):
    category_name_dict = {"03001627": "Chair", "04379243": "Table", "02747177": "Trash bin", "02818832": "Bed",
                      "02871439": "Bookshelf", "02933112": "Cabinet", "04256520": "Sofa", "02808440": "Bathtub", 
                      "03636649": "Lamp", "02773838": "Bag", "03938244": "Pillow", "03337140": "File"}

    return category_name_dict.get(cat_str, "Other")

def get_category_idx_verbose(cat_str):
    category_idx_dict_cls_verbose = {"03001627": 0, "04379243": 1, "02747177": 2, "02818832": 3,
                      "02871439": 4, "02933112": 5, "04256520": 6, "02808440": 7, 
                      "03636649": 8, "02773838": 9, "03938244": 10, "03337140": 11}
                      
    return category_idx_dict_cls_verbose.get(cat_str, 12) # "other": 12

def get_symmetric_idx(sym_str: str):
    symmetry_idx_dict = {"__SYM_NONE": 0, "__SYM_ROTATE_UP_2": 1, "__SYM_ROTATE_UP_4": 2,  "__SYM_ROTATE_UP_INF": 3, }
    return symmetry_idx_dict[sym_str]

def get_symmetric_name(sym_idx: int):
    symmetry_list = ["__SYM_NONE", "__SYM_ROTATE_UP_2", "__SYM_ROTATE_UP_4", "__SYM_ROTATE_UP_INF"]
    return symmetry_list[sym_idx]

