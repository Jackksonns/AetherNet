# -*- coding: utf-8 -*-
"""
Base Trainer - 模板方法模式实现
===============================
定义训练流程的模板，子类可重写特定步骤以实现不同的训练逻辑。

核心流程:
    1. setup() - 初始化配置、模型、优化器、数据集
    2. train_epoch() - 单个epoch的训练
    3. validate() - 验证评估
    4. save_checkpoint() / load_checkpoint() - 检查点管理

Usage:
    >>> from core import BaseTrainer
    >>> 
    >>> class SwinIRTrainer(BaseTrainer):
    >>>     def build_model(self):
    >>>         return SwinIR(**self.config['model'])
    >>>
    >>> trainer = SwinIRTrainer(config)
    >>> trainer.train()
"""

import os
import time
import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class BaseTrainer(ABC):
    """
    训练器基类（模板方法模式）
    
    定义训练的整体流程骨架，具体实现由子类完成。
    
    Attributes:
        config (dict): 训练配置
        device (torch.device): 训练设备
        model (nn.Module): 训练模型
        optimizer (Optimizer): 优化器
        scheduler (_LRScheduler): 学习率调度器
        train_loader (DataLoader): 训练数据加载器
        val_loader (DataLoader): 验证数据加载器
        logger (Logger): 日志记录器
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化训练器
        
        Args:
            config: 训练配置字典
        """
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # 将在setup()中初始化
        self.model: Optional[nn.Module] = None
        self.optimizer: Optional[Optimizer] = None
        self.scheduler: Optional[_LRScheduler] = None
        self.train_loader: Optional[DataLoader] = None
        self.val_loader: Optional[DataLoader] = None
        self.criterion: Optional[nn.Module] = None
        
        # 训练状态
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = 0.0
        
        # 日志
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # 输出目录
        self.output_dir = config.get('output_dir', './experiments')
        os.makedirs(self.output_dir, exist_ok=True)
    
    # ==================== 模板方法 ====================
    
    def train(self) -> None:
        """
        主训练循环（模板方法）
        
        定义训练的整体流程，不建议重写此方法。
        """
        self.setup()
        
        total_epochs = self.config.get('epochs', 100)
        
        self.logger.info(f"Starting training for {total_epochs} epochs...")
        self.on_train_start()
        
        for epoch in range(self.current_epoch, total_epochs):
            self.current_epoch = epoch
            
            # 训练一个epoch
            self.on_epoch_start(epoch)
            train_loss = self.train_epoch()
            self.on_epoch_end(epoch, train_loss)
            
            # 验证
            if self.should_validate(epoch):
                val_metric = self.validate()
                self.on_validation_end(epoch, val_metric)
                
                # 保存最佳模型
                if val_metric > self.best_metric:
                    self.best_metric = val_metric
                    self.save_checkpoint('best.pth')
            
            # 定期保存检查点
            if self.should_save_checkpoint(epoch):
                self.save_checkpoint(f'epoch_{epoch}.pth')
            
            # 更新学习率
            if self.scheduler is not None:
                self.scheduler.step()
        
        self.on_train_end()
        self.logger.info(f"Training completed. Best metric: {self.best_metric:.4f}")
    
    def setup(self) -> None:
        """
        初始化设置
        
        按顺序调用各个构建方法，子类可重写特定方法。
        """
        self.model = self.build_model()
        self.model = self.model.to(self.device)
        
        self.train_loader, self.val_loader = self.build_dataloaders()
        self.optimizer = self.build_optimizer()
        self.scheduler = self.build_scheduler()
        self.criterion = self.build_criterion()
        
        # 多GPU支持
        if torch.cuda.device_count() > 1 and self.config.get('use_ddp', False):
            self.model = nn.DataParallel(self.model)
        
        self.logger.info(f"Model: {self.model.__class__.__name__}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
    
    def train_epoch(self) -> float:
        """
        训练一个epoch
        
        Returns:
            平均训练损失
        """
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            loss = self.train_step(batch)
            total_loss += loss
            self.global_step += 1
            
            if batch_idx % self.config.get('log_interval', 100) == 0:
                self.logger.info(f"Epoch [{self.current_epoch}][{batch_idx}/{num_batches}] Loss: {loss:.4f}")
        
        return total_loss / num_batches
    
    def validate(self) -> float:
        """
        验证评估
        
        Returns:
            验证指标（如PSNR）
        """
        self.model.eval()
        total_metric = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                metric = self.validation_step(batch)
                total_metric += metric
                num_batches += 1
        
        avg_metric = total_metric / max(num_batches, 1)
        self.logger.info(f"Validation - Epoch [{self.current_epoch}] Metric: {avg_metric:.4f}")
        return avg_metric
    
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
    def build_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        构建数据加载器
        
        Returns:
            (train_loader, val_loader)
        """
        pass
    
    @abstractmethod
    def train_step(self, batch: Any) -> float:
        """
        单个训练步骤
        
        Args:
            batch: 一个batch的数据
            
        Returns:
            当前batch的损失值
        """
        pass
    
    @abstractmethod
    def validation_step(self, batch: Any) -> float:
        """
        单个验证步骤
        
        Args:
            batch: 一个batch的数据
            
        Returns:
            当前batch的验证指标
        """
        pass
    
    # ==================== 可选重写的方法 ====================
    
    def build_optimizer(self) -> Optimizer:
        """构建优化器，默认使用Adam"""
        lr = self.config.get('lr', 1e-4)
        weight_decay = self.config.get('weight_decay', 0)
        return torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
    
    def build_scheduler(self) -> Optional[_LRScheduler]:
        """构建学习率调度器，默认返回None"""
        scheduler_config = self.config.get('scheduler', None)
        if scheduler_config is None:
            return None
        
        scheduler_type = scheduler_config.get('type', 'step')
        if scheduler_type == 'step':
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 30),
                gamma=scheduler_config.get('gamma', 0.1)
            )
        elif scheduler_type == 'cosine':
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('epochs', 100)
            )
        return None
    
    def build_criterion(self) -> nn.Module:
        """构建损失函数，默认使用L1Loss"""
        return nn.L1Loss()
    
    def should_validate(self, epoch: int) -> bool:
        """判断是否需要验证"""
        val_interval = self.config.get('val_interval', 1)
        return (epoch + 1) % val_interval == 0
    
    def should_save_checkpoint(self, epoch: int) -> bool:
        """判断是否需要保存检查点"""
        save_interval = self.config.get('save_interval', 10)
        return (epoch + 1) % save_interval == 0
    
    # ==================== 钩子方法 ====================
    
    def on_train_start(self) -> None:
        """训练开始时的回调"""
        pass
    
    def on_train_end(self) -> None:
        """训练结束时的回调"""
        pass
    
    def on_epoch_start(self, epoch: int) -> None:
        """每个epoch开始时的回调"""
        pass
    
    def on_epoch_end(self, epoch: int, train_loss: float) -> None:
        """每个epoch结束时的回调"""
        pass
    
    def on_validation_end(self, epoch: int, val_metric: float) -> None:
        """验证结束时的回调"""
        pass
    
    # ==================== 检查点管理 ====================
    
    def save_checkpoint(self, filename: str) -> None:
        """
        保存检查点
        
        Args:
            filename: 检查点文件名
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        path = os.path.join(self.output_dir, filename)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path: str) -> None:
        """
        加载检查点
        
        Args:
            path: 检查点文件路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1
        self.global_step = checkpoint['global_step']
        self.best_metric = checkpoint['best_metric']
        
        if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.logger.info(f"Checkpoint loaded: {path} (epoch {checkpoint['epoch']})")
