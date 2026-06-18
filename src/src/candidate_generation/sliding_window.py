import numpy as np
import torch
from tqdm import tqdm

def sliding_window_3d(volume, patch_size=96, stride=48):
    D, H, W = volume.shape
    ps = patch_size

    for z in range(0, D - ps + 1, stride):
        for y in range(0, H - ps + 1, stride):
            for x in range(0, W - ps + 1, stride):
                patch = volume[z:z+ps, y:y+ps, x:x+ps]
                center = (z + ps//2, y + ps//2, x + ps//2)
                yield patch, center
