import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    """
    PatchDiscriminator for GANs, typically used in image-to-image translation tasks
    such as Pix2Pix. Instead of classifying the whole image as real or fake,
    it classifies each N×N patch, providing more detailed feedback to the generator.

    Args:
        in_channels (int): Number of channels in the input image (default=3 for RGB images).
                           Since input and target images are concatenated, the actual input
                           to the first layer will be in_channels * 2.
    """
    def __init__(self, in_channels=3):
        super(PatchDiscriminator, self).__init__()

        # Define the network architecture using a sequential container
        self.model = nn.Sequential(
            # First block: No normalization, only Conv + LeakyReLU
            self._block(in_channels * 2, 64, norm=False),

            # Subsequent blocks: Conv + BatchNorm + LeakyReLU
            self._block(64, 128),
            self._block(128, 256),

            # Padding before deeper convolution
            nn.ZeroPad2d(1),

            # Deeper convolution: increase channels to 512
            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            # Final padding + single channel output for real/fake decision
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)  # Output is a matrix of patch-level decisions
        )

    def _block(self, in_channels, out_channels, norm=True):
        """
        Helper function to define a convolutional block.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            norm (bool): Whether to include Batch Normalization.

        Returns:
            nn.Sequential: A block consisting of Conv2d, optional BatchNorm2d, and LeakyReLU.
        """
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        ]
        if norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def forward(self, input_image, target_image):
        """
        Forward pass of the PatchDiscriminator.

        Args:
            input_image (Tensor): The input image tensor of shape (B, C, H, W).
            target_image (Tensor): The target/generated image tensor of the same shape.

        Returns:
            Tensor: The patch-level prediction map of shape (B, 1, H', W').
        """
        # Concatenate input and target images along the channel dimension
        x = torch.cat([input_image, target_image], dim=1)
        return self.model(x)
