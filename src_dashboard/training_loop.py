import torch
from utils import apply_rhythmic_annealing

class TrainingLoop:
    def __init__(self, model, optimizer, loss_fn, data_loader, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.data_loader = data_loader
        self.device = device
        self.model.to(device)

    def train_epoch(self, rhythmic_annealing_strength=1.0, current_epoch=0):
        self.model.train()
        total_loss = 0

        for batch_X, batch_y in self.data_loader:
            batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(batch_X)
            loss = self.loss_fn(outputs, batch_y)
            loss.backward()
            self.optimizer.step()

            apply_rhythmic_annealing(self.model, rhythmic_annealing_strength, current_epoch)

            total_loss += loss.item()

        return total_loss / len(self.data_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                outputs = self.model(batch_X)
                loss = self.loss_fn(outputs, batch_y)
                total_loss += loss.item()

        return total_loss / len(val_loader)