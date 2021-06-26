
import torch
import torchvision
import torchvision.transforms as T
import argparse
import numpy as np
from PIL import Image

from models.deeplabv2 import DeepLabV2_ResNet101

COLORS = [(128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153), (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152), (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70), (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32), (0, 0, 0)]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type = str)
    parser.add_argument("--input_path", type = str)
    parser.add_argument("--output_path", type = str)
    parser.add_argument("--alpha", type=float, default=0.5)