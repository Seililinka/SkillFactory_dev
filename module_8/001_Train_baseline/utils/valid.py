import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import  accuracy_score


def valid(model: nn.Module,
         dataloader: DataLoader,
         criteria: nn.Module,
         device: str) -> float:

    avg_loss = 0
    avg_acc = 0
    model.eval()

    for i, (imgs, labels) in enumerate(dataloader):
        imgs = imgs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            out = model(imgs)
            loss = criteria(out, labels)

        out = torch.argmax(out, dim=1)
        avg_loss += round(loss.item(), 3)
        avg_acc += accuracy_score(labels.int().flatten().cpu(), out.int().flatten().cpu())

    return round(avg_loss / len(dataloader), 3), round(avg_acc  / len(dataloader), 3)
