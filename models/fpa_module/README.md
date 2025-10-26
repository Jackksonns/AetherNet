# FPA (Frequency Perception Attention) Module

This module implements the Frequency Perception Attention (FPA) proposed in the LKFN (Large Kernel Frequency-enhanced Network) model, accepted at CVPRW 2024.

## Overview

The FPA module enhances image restoration models by incorporating frequency-domain information into the attention mechanism. It performs Fast Fourier Transform (FFT) on input features, applies learnable transformations in the frequency domain, and then transforms back to the spatial domain to generate attention maps.

## Features

- Standalone implementation that can be easily integrated into other projects
- Compatible with the original LKFN implementation
- Supports configurable FFT normalization modes
- Efficient implementation leveraging PyTorch's native FFT operations

## Installation

Simply copy the `fpa_module` folder to your project directory.

## Usage

### Basic Usage

```python
from fpa_module import FPA

# Create FPA module with 64 input channels
fpa = FPA(embed_dim=64)

# Apply FPA to your features
# x shape: (batch_size, 64, height, width)
attended_features = fpa(x)
```

### As Part of a Larger Model

```python
import torch.nn as nn
from fpa_module import FPA

class YourImageRestorationModel(nn.Module):
    def __init__(self, in_channels=3, embed_dim=64):
        super().__init__()
        self.feature_extractor = nn.Conv2d(in_channels, embed_dim, 3, 1, 1)
        self.fpa = FPA(embed_dim=embed_dim)
        self.reconstruction = nn.Conv2d(embed_dim, in_channels, 3, 1, 1)
        
    def forward(self, x):
        features = self.feature_extractor(x)
        attended_features = self.fpa(features)
        output = self.reconstruction(attended_features)
        return output
```

## API Reference

### FPA(embed_dim, fft_norm="ortho")

Main class implementing the Frequency Perception Attention.

#### Parameters:
- `embed_dim` (int): Number of input channels
- `fft_norm` (str): FFT normalization mode. Options: "ortho", "forward", "backward", None. Default: "ortho"

### Attention(embed_dim, fft_norm="ortho")

The core attention mechanism used by FPA. Can be used independently if needed.

#### Parameters:
- `embed_dim` (int): Number of input channels
- `fft_norm` (str): FFT normalization mode. Options: "ortho", "forward", "backward", None. Default: "ortho"

## Integration with Existing Models

To integrate FPA into your existing image restoration models as a shallow feature extraction enhancement:

1. Import the FPA module:
   ```python
   from fpa_module import FPA
   ```

2. Add FPA after your initial feature extraction layers:
   ```python
   self.initial_conv = nn.Conv2d(3, 64, 3, 1, 1)
   self.fpa = FPA(embed_dim=64)
   ```

3. In your forward function:
   ```python
   features = self.initial_conv(x)
   attended_features = self.fpa(features)
   ```

## Citation

If you use this module in your research, please cite the original LKFN paper:

```
@InProceedings{zhao2024lkfn,
    author    = {Zhao, Junjun and Li, Zhentao and Liu, Hanzi and Qiu, Chenqi and Wang, Zenghui and Yi, Hongwei},
    title     = {LKFN: Large Kernel Frequency-enhanced Network for Efficient Single Image Super-Resolution},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW)},
    year      = {2024}
}
```