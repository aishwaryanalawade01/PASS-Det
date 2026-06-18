import torch
from torch.utils.data import DataLoader
from src.datasets.patch_dataset import LungPatchDataset
from src.models.stage1_cnn import Stage1CNN

device = "cuda"

# Dataset
ds = LungPatchDataset()
dl = DataLoader(ds, batch_size=2, shuffle=True, num_workers=2)

# Model
model = Stage1CNN().to(device)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

model.train()

for step, (x, y) in enumerate(dl):
    x = x.to(device)
    y = y.to(device).unsqueeze(1)

    logits = model(x)
    loss = criterion(logits, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if step % 50 == 0:
        torch.save(model.state_dict(), "stage1.pt")

    if step % 10 == 0:
        print(f"Step {step} | Loss {loss.item():.4f}")

    if step == 50:  # short run
        break

