import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.datasets.stage2_dataset_gt import Stage2DatasetGT
from src.models.stage1_cnn import Stage1CNN
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_STEPS = 500
BATCH_SIZE = 8

class Stage2Refiner(Stage1CNN):
    pass

ds = Stage2DatasetGT()
pos = sum(1 for s in ds.samples if s[2]==1)
neg = sum(1 for s in ds.samples if s[2]==0)

if pos == 0:
    print("ERROR: No positive samples. Check MATCH_RADIUS or GT file.")
    exit()

weights = [1.0/pos if s[2]==1 else 1.0/neg for s in ds.samples]
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
dl = DataLoader(ds, batch_size=BATCH_SIZE, sampler=sampler, num_workers=0)

model = Stage2Refiner().to(DEVICE)
model.load_state_dict(torch.load("stage1_gt.pt", map_location=DEVICE))
print("Warm-started from stage1_gt.pt")

pos_weight = torch.tensor([min(neg/max(pos,1), 20.0)]).to(DEVICE)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_STEPS)

model.train()
data_iter = iter(dl)
pbar = tqdm(total=MAX_STEPS, desc="Stage-2 GT Training")
best_loss = float("inf")

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
    scheduler.step()

    if step % 20 == 0:
        tqdm.write(f"Step {step:4d} | Loss {loss.item():.4f}")
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.state_dict(), "stage2_gt_best.pt")
    pbar.update(1)

pbar.close()
torch.save(model.state_dict(), "stage2_gt.pt")
print(f"Done. Best loss: {best_loss:.4f}")
print("Saved: stage2_gt.pt / stage2_gt_best.pt")
