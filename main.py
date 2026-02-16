import torch
import torch.nn as nn

from src.dataset import get_dataloaders
from src.model import SimpleCNN
from src.train import train_model

def main():
# === Device ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("using device:", device)

# === Data ===
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

# === Model ===
    model = SimpleCNN(num_classes=10)

# === Loss Function ===
    criterion= nn.CrossEntropyLoss()

# === Optimizer ===
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# === Train ===
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        num_epochs=5
    )



if __name__ == "__main__":
    main()

