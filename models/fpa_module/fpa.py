import torch
from torch import nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    """Attention module that implements frequency perception attention (FPA)"""

    def __init__(self, embed_dim, fft_norm="ortho"):
        """
        Initialize the Attention module.

        Args:
            embed_dim (int): The number of input channels.
            fft_norm (str): The normalization mode for FFT. Default: "ortho".
        """
        super(Attention, self).__init__()
        self.conv_layer1 = torch.nn.Conv2d(embed_dim, embed_dim // 2, 1, 1, 0)
        self.conv_layer2 = torch.nn.Conv2d(embed_dim // 2, embed_dim // 2, 1, 1, 0)
        self.conv_layer3 = torch.nn.Conv2d(embed_dim // 2, embed_dim, 1, 1, 0)
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.fft_norm = fft_norm

    def forward(self, x):
        """
        Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Output tensor after applying frequency perception attention.
        """
        fft_dim = (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        real = ffted.real + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.real))))
        )
        imag = ffted.imag + self.conv_layer3(
            self.relu(self.conv_layer2(self.relu(self.conv_layer1(ffted.imag))))
        )
        ffted = torch.complex(real, imag)

        ifft_shape_slice = x.shape[-2:]

        output = torch.fft.irfftn(
            ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm
        )

        return x * output


class FPA(nn.Module):
    """
    Frequency Perception Attention (FPA) Module
    This is a standalone module that can be used in other projects as a feature extraction component.
    """

    def __init__(self, embed_dim, fft_norm="ortho"):
        """
        Initialize the FPA module.

        Args:
            embed_dim (int): The number of input channels.
            fft_norm (str): The normalization mode for FFT. Default: "ortho".
        """
        super(FPA, self).__init__()
        self.atten = Attention(embed_dim, fft_norm)

    def forward(self, x):
        """
        Forward function.

        Args:
            x (Tensor): Input tensor with shape (B, C, H, W).

        Returns:
            Tensor: Output tensor after applying frequency perception attention.
        """
        return self.atten(x)