"""
Example usage of the FPA (Frequency Perception Attention) module.
This example demonstrates how to integrate FPA into a simple image restoration model.
"""

import torch
import torch.nn as nn
from fpa import FPA


class SimpleRestorationModel(nn.Module):
    """
    A simple example image restoration model using FPA for feature enhancement.
    """
    
    def __init__(self, in_channels=3, embed_dim=64):
        super(SimpleRestorationModel, self).__init__()
        # Initial feature extraction
        self.head = nn.Conv2d(in_channels, embed_dim, kernel_size=3, padding=1)
        
        # FPA module for frequency-enhanced feature attention
        self.fpa = FPA(embed_dim=embed_dim)
        
        # Reconstruction
        self.tail = nn.Conv2d(embed_dim, in_channels, kernel_size=3, padding=1)
        
        # Activation
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        # Extract initial features
        features = self.relu(self.head(x))
        
        # Apply frequency perception attention
        attended_features = self.fpa(features)
        
        # Reconstruct the image
        out = self.tail(attended_features)
        
        # Residual connection
        return x + out


def demo():
    """
    Demonstrate the usage of FPA module.
    """
    # Create model
    model = SimpleRestorationModel(in_channels=3, embed_dim=64)
    
    # Create sample input (batch_size=1, channels=3, height=64, width=64)
    sample_input = torch.randn(1, 3, 64, 64)
    
    # Forward pass
    output = model(sample_input)
    
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print("FPA module working correctly!")


if __name__ == "__main__":
    demo()