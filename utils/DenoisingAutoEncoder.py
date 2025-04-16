import torch
import torch.nn as nn
import torch.nn.functional as F


class DenoisingAutoencoder(nn.Module):
    def __init__(self, in_channels=3):
        super(DenoisingAutoencoder, self).__init__()
        
        # ----------> Encoder <----------
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=5, padding=2),  # (256x256x3) → (256x256x64)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),                  # → (128x128x64)

            nn.Conv2d(64, 128, kernel_size=3, padding=1),          # → (128x128x128)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0),                  # → (64x64x128)

            nn.Conv2d(128, 256, kernel_size=3, padding=1),         # → (64x64x256)
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, padding=0)                   # → (32x32x256)
        )

        # ----------> Decoder <----------
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3, padding=1),  # → (32x32x256)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                             # → (64x64x256)

            nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1),  # → (64x64x128)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                             # → (128x128x128)

            nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1),   # → (128x128x64)
            nn.ReLU(),
            nn.Upsample(scale_factor=2),                             # → (256x256x64)

            nn.ConvTranspose2d(64, 3, kernel_size=5, padding=2),     # → (256x256x3)
            nn.Sigmoid()                                             # Output range [0, 1]
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
