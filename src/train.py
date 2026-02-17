import torch
import os
os.makedirs("models", exist_ok=True)

def train_model(model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                device,
                num_epochs=10):
    
    best_val_loss = float("inf")
    patience = 2
    counter = 0 

    model.to(device)

    for epoch in range(num_epochs):

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("---------------------------------------------")

        # === TRAINING PHASE ===

        model.train()  # Set model to training mode

        train_loss = 0.0
        total_train_samples = 0

        for images, labels in train_loader:

            # Move data to device (CPU/GPU)
            images = images.to(device)
            labels = labels.to(device)

            # Reset gradients otherwhise it will sum all
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Compute loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()

            # Update weights
            optimizer.step()

            # Accumulate loss
            batch_size = images.size(0)
           # loss.item is the avarage of the batch
            train_loss += loss.item() * batch_size
            total_train_samples += batch_size

        average_train_loss = train_loss / total_train_samples

        print(f"Train Loss: {average_train_loss:.4f}")

        # ==== Validation ====

        model.eval()  # Set model to evaluation mode

        val_loss = 0.0
        total_val_samples = 0

        with torch.no_grad():  # Disable gradient computation

            for images, labels in val_loader:

                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)

                loss = criterion(outputs, labels)

                batch_size = images.size(0)
                val_loss += loss.item() * batch_size
                total_val_samples += batch_size

        average_val_loss = val_loss / total_val_samples

        # detect and save the best model
        if average_val_loss < best_val_loss:
            best_val_loss = average_val_loss
            counter = 0
            torch.save(model.state_dict(), "models/exp1_lr001_bc64.pth")
            print("Best model saved.")
        else:
            counter += 1
            print(f"No improvement. Counter: {counter}")

        if counter >= patience:
            print("Early stopping triggered.")
            break

        print(f"Validation Loss: {average_val_loss:.4f}")
