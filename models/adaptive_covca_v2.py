import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

class SimplifiedCVOCA(nn.Module):
    """
    简化的CVOCA核心模块，去除过于理想的光学模拟
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        
        # 简化版的复杂值卷积权重
        self.weight_real = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.weight_imag = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        
        # 初始化权重
        self._initialize_weights()
        
    def _initialize_weights(self):
        """初始化权重"""
        with torch.no_grad():
            if self.out_channels >= 4:
                # 示例Sobel-like核
                S = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32)
                ST = S.T
                
                # 简化版的核设计
                self.weight_real[0, 0] = S
                self.weight_imag[0, 0] = ST
                
                self.weight_real[1, 0] = S
                self.weight_imag[1, 0] = S
                
                self.weight_real[2, 0] = ST
                self.weight_imag[2, 0] = ST
                
                self.weight_real[3, 0] = ST
                self.weight_imag[3, 0] = S
    
    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        简化版的前向传播
        
        Args:
            x_real: 输入实部 [batch, channels, height, width]
            x_imag: 输入虚部 [batch, channels, height, width]
            
        Returns:
            output_real: 输出特征图实部 [batch, out_channels, height', width']
            output_imag: 输出特征图虚部 [batch, out_channels, height', width']
        """
        batch_size, channels, height, width = x_real.shape
        outputs_real = []
        outputs_imag = []
        
        # 对每个输出通道单独处理
        for out_ch in range(self.out_channels):
            channel_outputs_real = []
            channel_outputs_imag = []
            
            for in_ch in range(channels):
                # 简化版的复数卷积操作
                # Y = WR * XR - WI * XI + j * (WR * XI + WI * XR)
                real_part = F.conv2d(x_real[:, in_ch:in_ch+1], self.weight_real[out_ch, in_ch:in_ch+1], padding=1) - \
                           F.conv2d(x_imag[:, in_ch:in_ch+1], self.weight_imag[out_ch, in_ch:in_ch+1], padding=1)
                
                imag_part = F.conv2d(x_real[:, in_ch:in_ch+1], self.weight_imag[out_ch, in_ch:in_ch+1], padding=1) + \
                           F.conv2d(x_imag[:, in_ch:in_ch+1], self.weight_real[out_ch, in_ch:in_ch+1], padding=1)
                
                channel_outputs_real.append(real_part)
                channel_outputs_imag.append(imag_part)
            
            # 跨输入通道求和
            channel_real = sum(channel_outputs_real)
            channel_imag = sum(channel_outputs_imag)
            
            outputs_real.append(channel_real)
            outputs_imag.append(channel_imag)
        
        # 堆叠输出通道
        output_real = torch.cat(outputs_real, dim=1)
        output_imag = torch.cat(outputs_imag, dim=1)
        
        return output_real, output_imag


class AdaptiveCVOCA_V2(nn.Module):
    """
    适应低质量图像的CVOCA模块 V2
    改进的复数表示构建方法
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, 
                 use_denoising: bool = True, use_multiscale: bool = True):
        super().__init__()
        
        self.use_denoising = use_denoising
        self.use_multiscale = use_multiscale
        
        # 预处理的噪声抑制
        if use_denoising:
            self.denoise_real = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1)
            )
            self.denoise_imag = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 1, 3, padding=1)
            )
        
        # 多尺度特征提取
        if use_multiscale:
            self.multiscale_real = nn.ModuleList([
                nn.Conv2d(1, 8, 3, padding=1, dilation=1),
                nn.Conv2d(1, 8, 3, padding=2, dilation=2),
                nn.Conv2d(1, 8, 3, padding=3, dilation=3)
            ])
            self.multiscale_imag = nn.ModuleList([
                nn.Conv2d(1, 8, 3, padding=1, dilation=1),
                nn.Conv2d(1, 8, 3, padding=2, dilation=2), 
                nn.Conv2d(1, 8, 3, padding=3, dilation=3)
            ])
            # 融合多尺度特征
            self.fuse_multiscale = nn.Conv2d(24, 1, 1)
        
        # 简化的CVOCA核心（去除过于理想的光学模拟）
        self.cvoca_core = SimplifiedCVOCA(in_channels, out_channels, kernel_size)
        
        # 特征重要性模块
        self.feature_importance = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels * 2, out_channels // 2, 1),
            nn.ReLU(),
            nn.Conv2d(out_channels // 2, out_channels * 2, 1),
            nn.Sigmoid()
        )

    def build_complex_representation(self, x_normed: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        为低质量图像构建更鲁棒的复数表示
        改进版本：使用多尺度结构信息而非简单梯度
        """
        if x_normed.shape[1] == 3:
            # 使用亮度通道作为实部
            x_real = torch.mean(x_normed, dim=1, keepdim=True)
        else:
            x_real = x_normed
        
        # 噪声抑制
        if self.use_denoising:
            x_real_denoised = self.denoise_real(x_real)
        else:
            x_real_denoised = x_real
            
        # 改进的虚部构建：使用多尺度结构信息而非简单梯度
        if self.use_multiscale:
            # 多尺度结构信息
            scales = [1, 2, 4]
            imaginary_parts = []
            for scale in scales:
                # 下采样再上采样获取结构信息
                downsampled = F.interpolate(x_real_denoised, scale_factor=1/scale, mode='bilinear', align_corners=False)
                upsampled = F.interpolate(downsampled, size=x_real_denoised.shape[2:], mode='bilinear', align_corners=False)
                imaginary_parts.append(x_real_denoised - upsampled)  # 残差作为结构信息
            
            x_imag_multiscale = torch.cat(imaginary_parts, dim=1)
            # 融合多尺度结构信息
            x_imag = self.fuse_multiscale(x_imag_multiscale)
            
            if self.use_denoising:
                x_imag = self.denoise_imag(x_imag)
        else:
            # 基础梯度特征
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                                  dtype=torch.float32, device=x_normed.device).view(1, 1, 3, 3)
            sobel_y = sobel_x.transpose(2, 3)
            grad_x = F.conv2d(x_real_denoised, sobel_x, padding=1)
            grad_y = F.conv2d(x_real_denoised, sobel_y, padding=1)
            x_imag = torch.sqrt(grad_x**2 + grad_y**2 + 1e-8)
            
            if self.use_denoising:
                x_imag = self.denoise_imag(x_imag)
        
        return x_real_denoised, x_imag

    def forward(self, x_real: torch.Tensor, x_imag: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            x_real: 输入实部 [batch, channels, height, width]
            x_imag: 输入虚部 [batch, channels, height, width]
            
        Returns:
            output_real: 输出特征图实部 [batch, out_channels, height', width']
            output_imag: 输出特征图虚部 [batch, out_channels, height', width']
        """
        real_out, imag_out = self.cvoca_core(x_real, x_imag)
        
        # 应用特征重要性重标定
        combined_features = torch.cat([real_out, imag_out], dim=1)
        importance_weights = self.feature_importance(combined_features)
        weighted_features = combined_features * importance_weights
        real_out, imag_out = torch.chunk(weighted_features, 2, dim=1)
        
        return real_out, imag_out