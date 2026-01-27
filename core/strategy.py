# -*- coding: utf-8 -*-
"""
Degradation Strategy - 策略模式实现
===================================
定义图像退化策略的抽象接口和具体实现，用于训练数据的退化处理。

支持的退化类型：
- ClassicalSRDegradation: 经典超分辨率退化（双三次下采样）
- ColorDenoiseDegradation: 彩色图像去噪退化（高斯噪声）
- JPEGCARDegradation: JPEG压缩伪影退化

Usage:
    >>> from core import ClassicalSRDegradation
    >>> 
    >>> degradation = ClassicalSRDegradation(scale=4)
    >>> lr_image = degradation.apply(hr_image)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.nn.functional as F


class DegradationStrategy(ABC):
    """
    退化策略抽象基类（策略模式）
    
    定义图像退化的通用接口，所有具体退化策略必须实现 apply 方法。
    """
    
    @abstractmethod
    def apply(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        应用退化操作
        
        Args:
            image: 高质量图像（HR/Clean）
            
        Returns:
            退化后的低质量图像（LR/Degraded）
        """
        pass
    
    @abstractmethod
    def get_config(self) -> dict:
        """
        获取退化策略的配置信息
        
        Returns:
            配置字典
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.get_config()})"


class ClassicalSRDegradation(DegradationStrategy):
    """
    经典超分辨率退化策略
    
    使用双三次插值进行下采样，模拟经典超分辨率任务的退化过程。
    
    Attributes:
        scale (int): 下采样倍率
        mode (str): 插值模式
    """
    
    def __init__(self, scale: int = 4, mode: str = 'bicubic'):
        """
        初始化经典SR退化策略
        
        Args:
            scale: 下采样倍率，默认为4
            mode: 插值模式，可选 'bicubic', 'bilinear', 'nearest'
        """
        self.scale = scale
        self.mode = mode
    
    def apply(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        应用双三次下采样
        
        Args:
            image: HR图像，形状为 (H, W, C) 或 (B, C, H, W)
            
        Returns:
            LR图像
        """
        if isinstance(image, np.ndarray):
            return self._apply_numpy(image)
        elif isinstance(image, torch.Tensor):
            return self._apply_tensor(image)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _apply_numpy(self, image: np.ndarray) -> np.ndarray:
        """NumPy实现"""
        import cv2
        h, w = image.shape[:2]
        new_h, new_w = h // self.scale, w // self.scale
        
        interpolation = {
            'bicubic': cv2.INTER_CUBIC,
            'bilinear': cv2.INTER_LINEAR,
            'nearest': cv2.INTER_NEAREST,
        }.get(self.mode, cv2.INTER_CUBIC)
        
        return cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    def _apply_tensor(self, image: torch.Tensor) -> torch.Tensor:
        """PyTorch实现"""
        if image.dim() == 3:  # (C, H, W)
            image = image.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        _, _, h, w = image.shape
        new_h, new_w = h // self.scale, w // self.scale
        
        result = F.interpolate(image, size=(new_h, new_w), mode=self.mode, 
                               align_corners=False if self.mode != 'nearest' else None)
        
        return result.squeeze(0) if squeeze else result
    
    def get_config(self) -> dict:
        return {'scale': self.scale, 'mode': self.mode}


class ColorDenoiseDegradation(DegradationStrategy):
    """
    彩色图像去噪退化策略
    
    添加高斯噪声，模拟图像去噪任务的退化过程。
    
    Attributes:
        sigma (float): 噪声标准差 (0-255 范围)
        sigma_range (tuple): 噪声强度范围（用于随机采样）
    """
    
    def __init__(self, sigma: float = 25, sigma_range: Optional[Tuple[float, float]] = None):
        """
        初始化去噪退化策略
        
        Args:
            sigma: 固定噪声强度，默认为25
            sigma_range: 噪声强度范围，如 (0, 50)，设置后将随机采样
        """
        self.sigma = sigma
        self.sigma_range = sigma_range
    
    def apply(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        添加高斯噪声
        
        Args:
            image: 干净图像
            
        Returns:
            带噪声的图像
        """
        # 确定噪声强度
        if self.sigma_range is not None:
            sigma = np.random.uniform(self.sigma_range[0], self.sigma_range[1])
        else:
            sigma = self.sigma
        
        if isinstance(image, np.ndarray):
            return self._apply_numpy(image, sigma)
        elif isinstance(image, torch.Tensor):
            return self._apply_tensor(image, sigma)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
    
    def _apply_numpy(self, image: np.ndarray, sigma: float) -> np.ndarray:
        """NumPy实现"""
        noise = np.random.randn(*image.shape).astype(np.float32) * (sigma / 255.)
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.
        noisy = image + noise
        return np.clip(noisy, 0., 1.)
    
    def _apply_tensor(self, image: torch.Tensor, sigma: float) -> torch.Tensor:
        """PyTorch实现"""
        noise = torch.randn_like(image) * (sigma / 255.)
        noisy = image + noise
        return torch.clamp(noisy, 0., 1.)
    
    def get_config(self) -> dict:
        return {'sigma': self.sigma, 'sigma_range': self.sigma_range}


class JPEGCARDegradation(DegradationStrategy):
    """
    JPEG压缩伪影退化策略
    
    通过JPEG压缩模拟压缩伪影，用于JPEG伪影去除任务。
    
    Attributes:
        quality (int): JPEG质量因子 (0-100)
        quality_range (tuple): 质量范围（用于随机采样）
    """
    
    def __init__(self, quality: int = 40, quality_range: Optional[Tuple[int, int]] = None):
        """
        初始化JPEG退化策略
        
        Args:
            quality: 固定JPEG质量，默认为40
            quality_range: 质量范围，如 (10, 40)，设置后将随机采样
        """
        self.quality = quality
        self.quality_range = quality_range
    
    def apply(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """
        应用JPEG压缩
        
        Args:
            image: 干净图像
            
        Returns:
            JPEG压缩后的图像
        """
        import cv2
        
        # 确定质量
        if self.quality_range is not None:
            quality = np.random.randint(self.quality_range[0], self.quality_range[1] + 1)
        else:
            quality = self.quality
        
        is_tensor = isinstance(image, torch.Tensor)
        if is_tensor:
            device = image.device
            image = image.cpu().numpy()
            if image.ndim == 3 and image.shape[0] in [1, 3]:  # (C, H, W)
                image = image.transpose(1, 2, 0)
        
        # 转换为uint8进行JPEG压缩
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
        
        # JPEG编码和解码
        encode_param = [cv2.IMWRITE_JPEG_QUALITY, quality]
        _, encoded = cv2.imencode('.jpg', image, encode_param)
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR if image.ndim == 3 else cv2.IMREAD_GRAYSCALE)
        
        # 转回float32
        result = decoded.astype(np.float32) / 255.
        
        if is_tensor:
            if result.ndim == 3:
                result = result.transpose(2, 0, 1)
            result = torch.from_numpy(result).to(device)
        
        return result
    
    def get_config(self) -> dict:
        return {'quality': self.quality, 'quality_range': self.quality_range}


# ==================== 策略工厂函数 ====================

def create_degradation(task: str, **kwargs) -> DegradationStrategy:
    """
    根据任务类型创建退化策略
    
    Args:
        task: 任务类型，可选 'classical_sr', 'color_dn', 'jpeg_car'
        **kwargs: 传递给退化策略的参数
        
    Returns:
        对应的退化策略实例
        
    Raises:
        ValueError: 如果任务类型不支持
    """
    strategies = {
        'classical_sr': ClassicalSRDegradation,
        'color_dn': ColorDenoiseDegradation,
        'jpeg_car': JPEGCARDegradation,
    }
    
    if task not in strategies:
        raise ValueError(f"Unknown task: {task}. Available: {list(strategies.keys())}")
    
    return strategies[task](**kwargs)
