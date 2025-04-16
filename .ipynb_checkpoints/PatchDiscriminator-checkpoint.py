import torch
import torch.nn as nn


class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()
        self.model = nn.Sequential(
            self._block(in_channels * 2, 64, norm=False),
            self._block(64, 128),
            self._block(128, 256),
            nn.ZeroPad2d(1),
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)
        )

    def _block(self, in_channels, out_channels, norm=True):
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, input_image, target_image):
        x = torch.cat([input_image, target_image], dim=1) 
        return self.model(x)
