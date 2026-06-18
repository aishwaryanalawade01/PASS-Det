import torch
from torch.utils.data import DataLoader
from src.datasets.patch_dataset import LungPatchDataset
from src.models.stage1_detector import UNet3D
from src.losses.focal_loss import focal_loss

device = "cuda"

ds = LungPatchDataset("/home/AishwaryaNalawade/data/npy")
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)

model = UNet3D().to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

x, _ = next(iter(dl))
x = x.to(device)

# Dummy GT heatmap (all zeros)
gt = torch.zeros_like(x[:, 0])

pred = model(x)

loss = focal_loss(pred.squeeze(1), gt)
loss.backward()
optim.step()

print("Sanity loss:", loss.item())
