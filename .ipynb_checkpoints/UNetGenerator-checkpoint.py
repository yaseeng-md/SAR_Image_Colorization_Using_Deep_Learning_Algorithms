import torch
import torch.nn as nn


class UNetBlockDown(nn.Module):
    def __init__(self, in_channels, out_channels, apply_batchnorm=True):
        super(UNetBlockDown, self).__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetBlockUp(nn.Module):
    def __init__(self, in_channels, out_channels, apply_batchnorm=True, apply_dropout=False):
        super(UNetBlockUp, self).__init__()
        layers = [nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)]
        if apply_batchnorm:
            layers.append(nn.BatchNorm2d(out_channels))
        if apply_dropout:
            layers.append(nn.Dropout(0.5))
        layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        self.down1 = UNetBlockDown(in_channels, 64, apply_batchnorm=False)
        self.down2 = UNetBlockDown(64, 128)
        self.down3 = UNetBlockDown(128, 256)
        self.down4 = UNetBlockDown(256, 512)
        self.down5 = UNetBlockDown(512, 512)
        self.down6 = UNetBlockDown(512, 512)
        self.down7 = UNetBlockDown(512, 512)
        self.down8 = UNetBlockDown(512, 512)

        self.up1 = UNetBlockUp(512, 512, apply_dropout=True)
        self.up2 = UNetBlockUp(1024, 512, apply_dropout=True)
        self.up3 = UNetBlockUp(1024, 512, apply_dropout=True)
        self.up4 = UNetBlockUp(1024, 512)
        self.up5 = UNetBlockUp(1024, 256)
        self.up6 = UNetBlockUp(512, 128)
        self.up7 = UNetBlockUp(256, 64)

        self.final = nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # Encoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        # Decoder with skip connections
        u1 = self.up1(d8)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))

        output = self.final(torch.cat([u7, d1], dim=1))
        return self.tanh(output)
