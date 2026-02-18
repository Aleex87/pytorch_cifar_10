import torch
import torch.nn as nn


class SimpleCNN(nn.Module):
   
    def __init__(self, num_classes: int = 10):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x32x32
            # in_channels = 3 => RGB
            # out_channels = 32 => we learn 32 different filters
            # kernel_size = 3 => filter 3x3
            # padding = 1 => keep same dimention in the space (32x32)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            # MaxPool2d(kernel_size =2)  downsampling from 3 to 2 reduce the dimention 
            # reduce complexiti
            nn.MaxPool2d(kernel_size=2),  # 32x32x32 -> 32x16x16

            # Block 2: 32x16x16 -> 64x16x16
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),  # 64x16x16 -> 64x8x8
        )

        # After two pooling operations, spatial size is 8x8.
        # Channels are 64. So total features = 64 * 8 * 8 = 4096
        self.classifier = nn.Sequential(
            nn.Flatten(),                  # (B, 64, 8, 8) -> (B, 4096)
            nn.Linear(64 * 8 * 8, 128),     # (B, 4096) -> (B, 128)
            nn.ReLU(),
            nn.Linear(128, num_classes)     # (B, 128) -> (B, 10)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Forward pass of the network.
        
        x = self.features(x)
        x = self.classifier(x)
        return x
