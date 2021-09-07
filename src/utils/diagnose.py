# Copyright: Chris Choy @ Nvidia

import sys
import os
import platform
import subprocess


def parse_nvidia_smi():
    sp = subprocess.Popen(
        ["nvidia-smi", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    out_dict = dict()
    for item in sp.communicate()[0].decode("utf-8").split("\n"):
        if item.count(":") == 1:
            key, val = [i.strip() for i in item.split(":")]
            out_dict[key] = val
    return out_dict


def print_diagnostics():
    print("==========System==========")
    print(platform.platform())
    os.system("cat /etc/lsb-release")
    print(sys.version)

    print("==========Python===========")
    print (sys.version)
    print (sys.version_info)

    print("==========Pytorch==========")
    try:
        import torch

        print(torch.__version__)
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    except ImportError:
        print("torch not installed")

    print("==========NVIDIA-SMI==========")
    os.system("which nvidia-smi")
    for k, v in parse_nvidia_smi().items():
        if "version" in k.lower():
            print(k, v)

    print("==========NVCC==========")
    os.system("which nvcc")
    os.system("nvcc --version")

    print("==========CC==========")
    CC = "c++"
    if "CC" in os.environ or "CXX" in os.environ:
        # distutils only checks CC not CXX
        if "CXX" in os.environ:
            os.environ["CC"] = os.environ["CXX"]
            CC = os.environ["CXX"]
        else:
            CC = os.environ["CC"]
        print(f"CC={CC}")
    os.system(f"which {CC}")
    os.system(f"{CC} --version")


if __name__ == "__main__":
    print_diagnostics()