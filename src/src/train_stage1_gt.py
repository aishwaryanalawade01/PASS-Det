import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.datasets.patch_dataset_gt import GTLungPatchDataset
from src.models.stage1_cnn import Stage1CNN
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 1000
BATCH_SIZE = 8

ds = GTLungPatchDataset()
labels = [s[2] for s in ds.samples]
n_pos = sum(labels)
n_neg = len(labels) - n_pos
weights = [1.0/n_pos if l==1 else 1.0/n_neg for l in labels]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

model = Stage1CNN().to(DEVICE)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

print(f"Retraining Stage-1 with TRUE GT nodule coordinates...")
model.train()
data_iter = iter(dl)
pbar = tqdm(total=MAX_STEPS, desc="Stage-1 GT Retrain")

for step in range(MAX_STEPS):
    try:
        x, y = next(data_iter)
    except StopIteration:
        data_iter = iter(dl)
        x, y = next(data_iter)

    x, y = x.to(DEVICE), y.to(DEVICE).unsqueeze(1)
    loss = criterion(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 20 == 0:
        tqdm.write(f"Step {step:4d} | Loss {loss.item():.4f}")
    if step % 200 == 0 and step > 0:
        torch.save(model.state_dict(), "stage1_gt.pt")
    pbar.update(1)

pbar.close()
torch.save(model.state_dict(), "stage1_gt.pt")
print("Done. Saved: stage1_gt.pt")
