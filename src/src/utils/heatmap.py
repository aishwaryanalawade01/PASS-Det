import torch
import numpy as np

def draw_gaussian_3d(heatmap, center, sigma=2):
    z, y, x = center
    zz, yy, xx = torch.meshgrid(
        torch.arange(heatmap.shape[0]),
        torch.arange(heatmap.shape[1]),
        torch.arange(heatmap.shape[2]),
        indexing="ij"
    )

    gaussian = torch.exp(
        -((zz - z)**2 + (yy - y)**2 + (xx - x)**2) / (2 * sigma**2)
    )

    heatmap = torch.maximum(heatmap, gaussian)
    return heatmap
