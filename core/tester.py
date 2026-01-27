# -*- coding: utf-8 -*-
"""
Base Tester - 模板方法模式实现
==============================
定义测试/推理流程的模板，子类可重写特定步骤以实现不同的测试逻辑。

核心流程:
    1. setup() - 加载模型和数据
    2. test() - 执行测试
    3. evaluate() - 计算评价指标
    4. save_results() - 保存结果

Usage:
    >>> from core import BaseTester
    >>> 
    >>> class SwinIRTester(BaseTester):
    >>>     def build_model(self):
    >>>         return SwinIR(**self.config['model'])
    >>>
    >>> tester = SwinIRTester(config, checkpoint_path='best.pth')
    >>> results = tester.test()
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class BaseTester(ABC):
    """
    测试器基类（模板方法模式）
    
    定义测试的整体流程骨架，具体实现由子类完成。
    
    Attributes:
        config (dict): 测试配置
        device (torch.device): 测试设备
        model (nn.Module): 测试模型
        test_loader (DataLoader): 测试数据加载器
        logger (Logger): 日志记录器
    """
    
    def __init__(self, config: Dict[str, Any], checkpoint_path: Optional[str] = None):
        """
        初始化测试器
        
        Args:
            config: 测试配置字典
            checkpoint_path: 预训练权重路径
        """
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 将在setup()中初始化
        self.model: Optional[nn.Module] = None
        self.test_loader: Optional[DataLoader] = None
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 输出目录
        self.output_dir = config.get('output_dir', './results')
        os.makedirs(self.output_dir, exist_ok=True)
    
    # ==================== 模板方法 ====================
    
    def test(self) -> Dict[str, float]:
        """
        主测试流程（模板方法）
        
        Returns:
            评价指标字典
        """
        self.setup()
        
        self.logger.info("Starting testing...")
        self.on_test_start()
        
        self.model.eval()
        all_results = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.test_loader):
                result = self.test_step(batch)
                all_results.append(result)
                
                if batch_idx % self.config.get('log_interval', 10) == 0:
                    self.logger.info(f"Processing [{batch_idx}/{len(self.test_loader)}]")
        
        # 汇总评价指标
        metrics = self.aggregate_metrics(all_results)
        
        self.on_test_end(metrics)
        self.logger.info(f"Testing completed. Metrics: {metrics}")
        
        return metrics
    
    def setup(self) -> None:
        """
        初始化设置
        """
        self.model = self.build_model()
        self.model = self.model.to(self.device)
        
        # 加载预训练权重
        if self.checkpoint_path is not None:
            self.load_checkpoint(self.checkpoint_path)
        
        self.test_loader = self.build_dataloader()
        
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Test samples: {len(self.test_loader.dataset)}")
    
    # ==================== 需要子类实现的方法 ====================
    
    @abstractmethod
    def build_model(self) -> nn.Module:
        """
        构建模型
        
        Returns:
            模型实例
        """
        pass
    
    @abstractmethod
    def build_dataloader(self) -> DataLoader:
        """
        构建测试数据加载器
        
        Returns:
            test_loader
        """
        pass
    
    @abstractmethod
    def test_step(self, batch: Any) -> Dict[str, Any]:
        """
        单个测试步骤
        
        Args:
            batch: 一个batch的数据
            
        Returns:
            包含预测结果和指标的字典
        """
        pass
    
    # ==================== 可选重写的方法 ====================
    
    def aggregate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        汇总所有batch的评价指标
        
        Args:
            results: 所有batch的结果列表
            
        Returns:
            汇总后的指标字典
        """
        if not results:
            return {}
        
        # 默认实现：计算所有数值指标的平均值
        metrics = {}
        for key in results[0].keys():
            if isinstance(results[0][key], (int, float)):
                values = [r[key] for r in results]
                metrics[key] = sum(values) / len(values)
        
        return metrics
    
    def save_results(self, outputs: torch.Tensor, filenames: List[str]) -> None:
        """
        保存输出结果
        
        Args:
            outputs: 模型输出张量 (B, C, H, W)
            filenames: 对应的文件名列表
        """
        import cv2
        
        outputs = outputs.cpu().numpy()
        
        for i, filename in enumerate(filenames):
            img = outputs[i]
            
            # (C, H, W) -> (H, W, C)
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = img.transpose(1, 2, 0)
            
            # 转换为uint8
            img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            
            # BGR -> RGB (OpenCV)
            if img.ndim == 3 and img.shape[2] == 3:
                img = img[:, :, ::-1]
            
            save_path = os.path.join(self.output_dir, filename)
            cv2.imwrite(save_path, img)
    
    # ==================== 钩子方法 ====================
    
    def on_test_start(self) -> None:
        """测试开始时的回调"""
        pass
    
    def on_test_end(self, metrics: Dict[str, float]) -> None:
        """测试结束时的回调"""
        pass
    
    # ==================== 检查点管理 ====================
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载检查点
        
        Args:
            path: 检查点文件路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        # 支持多种检查点格式
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        else:
            state_dict = checkpoint
        
        # 处理DataParallel的module.前缀
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        self.model.load_state_dict(new_state_dict, strict=False)
        self.logger.info(f"Checkpoint loaded: {path}")


# ==================== 评价指标工具函数 ====================

def calculate_psnr(img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
    """
    计算PSNR
    
    Args:
        img1: 图像1 (H, W, C) 或 (H, W)，范围[0, 255]
        img2: 图像2
        border: 边界裁剪像素数
        
    Returns:
        PSNR值
    """
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(255.0 / np.sqrt(mse))


def calculate_ssim(img1: np.ndarray, img2: np.ndarray, border: int = 0) -> float:
    """
    计算SSIM
    
    Args:
        img1: 图像1 (H, W, C) 或 (H, W)，范围[0, 255]
        img2: 图像2
        border: 边界裁剪像素数
        
    Returns:
        SSIM值
    """
    from scipy import ndimage
    
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    kernel = np.ones((11, 11)) / 121
    
    mu1 = ndimage.convolve(img1, kernel)
    mu2 = ndimage.convolve(img2, kernel)
    
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2
    
    sigma1_sq = ndimage.convolve(img1 ** 2, kernel) - mu1_sq
    sigma2_sq = ndimage.convolve(img2 ** 2, kernel) - mu2_sq
    sigma12 = ndimage.convolve(img1 * img2, kernel) - mu1_mu2
    
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(np.mean(ssim_map))
