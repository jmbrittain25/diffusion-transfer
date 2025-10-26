from abc import abstractmethod
from tqdm import tqdm


class BaseModelTrainer:

    def __init__(self):
        super().__init__()

    def train(self, model, loader, epochs, steps, loss_models):
        for epoch in tqdm(range(epochs), desc="Training"):
            train_loss = self.train_epoch(model, loader, optimizer, betas, device)
            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}")

    @abstractmethod
    def train_epoch():
        return

    @abstractmethod
    def evaluate(self, model, loader):
        return
