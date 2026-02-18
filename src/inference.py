import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from src.model import SimpleCNN


# Configuration

MODEL_PATH = "models/final_best_model.pth" 
IMAGE_FOLDER = "inference_images"

CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]

# Image Transform

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD)
])

# Load Model

def load_model(device):
    model = SimpleCNN(num_classes=10)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model

# Predict Single Image

def predict_image(model, image_path, device):
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception:
        print(f"[ERROR] Could not open file: {image_path}")
        return

    image = transform(image)
    image = image.unsqueeze(0) 
    # unsquese(0) add a bath dimention ay position 0 
    # which is required because the model expects batched input.

    with torch.no_grad():
        output = model(image.to(device))
        probabilities = F.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probabilities, 1)

    class_name = CLASS_NAMES[predicted_class.item()]
    confidence_value = confidence.item()

    print(f"Image: {image_path.name}")
    print(f"Predicted class: {class_name}")
    print(f"Confidence: {confidence_value:.4f}")
    print("-" * 40)

# Main

def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = load_model(device)

    image_dir = Path(IMAGE_FOLDER)

    if not image_dir.exists():
        print(f"[ERROR] Folder '{IMAGE_FOLDER}' does not exist.")
        return

    valid_extensions = [".png", ".jpg", ".jpeg"]

    images_found = False

    for file in image_dir.iterdir():
        if file.suffix.lower() in valid_extensions:
            images_found = True
            predict_image(model, file, device)

    if not images_found:
        print("No valid images found in the folder.")


if __name__ == "__main__":
    main()
