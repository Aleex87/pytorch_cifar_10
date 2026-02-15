from src.dataset import get_dataloaders

def main():
    train_loader, val_loader, test_loader = get_dataloaders(batch_size=32)

    images, labels = next(iter(train_loader))
    print(images.shape)   # expected: torch.Size([32, 3, 32, 32])
    print(labels.shape)   # expected: torch.Size([32])

if __name__ == "__main__":
    main()

