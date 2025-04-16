import torch
import torch.nn as nn

class UNetBlockDown(nn.Module):
    """
    A single downsampling block for the U-Net encoder.
    Applies Conv2D -> (optional) BatchNorm -> LeakyReLU
    """
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(UNetBlockDown, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetBlockUp(nn.Module):
    """
    A single upsampling block for the U-Net decoder.
    Applies ConvTranspose2D -> BatchNorm -> ReLU -> (optional Dropout)
    Then concatenates with the corresponding encoder feature map.
    """
    def __init__(self, in_channels, out_channels, apply_dropout=False):
        super(UNetBlockUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.block(x)
        x = torch.cat([x, skip_input], dim=1)  # Concatenate with skip connection along channel dimension
        return x


class UNetGenerator(nn.Module):
    """
    U-Net Generator for image-to-image tasks (e.g., colorization, segmentation).
    Encoder-decoder architecture with skip connections between mirrored layers.
    
    Args:
        in_channels (int): Number of input channels (e.g., 1 for grayscale)
        out_channels (int): Number of output channels (e.g., 3 for RGB)
    """
    def __init__(self, in_channels=1, out_channels=3):
        super(UNetGenerator, self).__init__()

        # Downsampling path (Encoder)
        self.down1 = UNetBlockDown(in_channels, 64, apply_batchnorm=False)  # No BN in the first layer
        self.down2 = UNetBlockDown(64, 128)
        self.down3 = UNetBlockDown(128, 256)
        self.down4 = UNetBlockDown(256, 512)
        self.down5 = UNetBlockDown(512, 512)
        self.down6 = UNetBlockDown(512, 512)
        self.down7 = UNetBlockDown(512, 512)
        self.down8 = UNetBlockDown(512, 512, apply_batchnorm=False)  # Bottleneck, no BN

        # Upsampling path (Decoder)
        self.up1 = UNetBlockUp(512, 512, apply_dropout=True)
        self.up2 = UNetBlockUp(1024, 512, apply_dropout=True)
        self.up3 = UNetBlockUp(1024, 512, apply_dropout=True)
        self.up4 = UNetBlockUp(1024, 512)
        self.up5 = UNetBlockUp(1024, 256)
        self.up6 = UNetBlockUp(512, 128)
        self.up7 = UNetBlockUp(256, 64)

        # Final layer: Transpose convolution to return to original resolution
        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()  # Output values in [-1, 1] range
        )

    def forward(self, x):
        # Encoder forward pass (store all outputs for skip connections)
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder forward pass with skip connections
        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        # Final output layer
        return self.final(u7)
