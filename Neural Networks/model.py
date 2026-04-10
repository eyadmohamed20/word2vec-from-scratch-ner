# model.py
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

from data import NUM_CLASSES


class FeedForwardNER(nn.Module):
    """
    Feed-Forward NER model with a configurable number of hidden layers.

    Parameters
    ----------
    input_dim    : embed_dim * (2*window+1)  — context window input
    hidden_dims  : list of hidden layer sizes, e.g. [128] or [256, 128]
    num_classes  : number of NER tags
    dropout      : dropout rate applied after each ReLU
    """

    def __init__(self, input_dim: int, hidden_dims: list,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        layers = []
        in_dim = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_batch), y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def evaluate(model, X_t, y_t, device, batch_size=512):
    from torch.utils.data import DataLoader, TensorDataset
    model.eval()
    preds = []
    with torch.no_grad():
        for X_batch, _ in DataLoader(TensorDataset(X_t, y_t), batch_size=batch_size):
            preds.append(model(X_batch.to(device)).argmax(dim=1).cpu())
    preds = torch.cat(preds).numpy()
    y_np  = y_t.numpy()
    return {
        "accuracy":  accuracy_score (y_np, preds),
        "precision": precision_score(y_np, preds, average="macro", zero_division=0),
        "recall":    recall_score   (y_np, preds, average="macro", zero_division=0),
        "f1":        f1_score       (y_np, preds, average="macro", zero_division=0),
        "report":    classification_report(y_np, preds, zero_division=0),
    }