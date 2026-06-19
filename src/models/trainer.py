import numpy as np
import torch
from src.config.config import DEVICE


class EarlyStopping:
    def __init__(self, patience: int = 5, verbose: bool = False,
                 delta: float = 1e-4, path: str = 'best_model.pth'):
        self.patience   = patience
        self.verbose    = verbose
        self.delta      = delta
        self.path       = path
        self.counter    = 0
        self.best_score = None
        self.early_stop = False
        self.auc_max    = -np.inf

    def __call__(self, auc: float, model: torch.nn.Module) -> None:
        if self.best_score is None:
            self.best_score = auc
            self._save(auc, model)
        elif auc < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'  EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = auc
            self._save(auc, model)
            self.counter = 0

    def _save(self, auc: float, model: torch.nn.Module) -> None:
        if self.verbose:
            print(f'  AUC mejoró ({self.auc_max:.4f} → {auc:.4f}). Guardando modelo.')
        torch.save(model.state_dict(), self.path)
        self.auc_max = auc


def run_one_epoch_train(model, loader, criterion, optimizer) -> float:
    model.train()
    total_loss = 0.0
    for inputs, targets in loader:
        inputs  = inputs.to(DEVICE)
        targets = targets.to(DEVICE).float()
        optimizer.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def run_one_epoch_val(model, loader, criterion, evaluator):
    model.eval()
    y_score    = torch.tensor([]).to(DEVICE)
    total_loss = 0.0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs  = inputs.to(DEVICE)
            targets = targets.to(DEVICE).float()
            outputs = model(inputs)
            total_loss += criterion(outputs, targets).item()
            y_score = torch.cat((y_score, torch.sigmoid(outputs)), dim=0)
    auc, acc = evaluator.evaluate(y_score.cpu().numpy())
    return total_loss / len(loader), auc, acc
