# -*- coding: utf-8 -*-
"""
Model Factory - 工厂模式实现
============================
提供统一的模型创建接口，支持通过配置字典或命令行参数创建模型实例。

Usage:
    >>> from core import ModelFactory
    >>> 
    >>> # 方式1：通过配置字典创建
    >>> config = {'model_name': 'SwinIR', 'upscale': 4, 'in_chans': 3, ...}
    >>> model = ModelFactory.create(config)
    >>>
    >>> # 方式2：通过名称和参数创建
    >>> model = ModelFactory.create_by_name('SwinIR', upscale=4, in_chans=3)
"""

from typing import Any, Dict, Optional, Union
import torch.nn as nn

from .registry import MODEL_REGISTRY


class ModelFactory:
    """
    模型工厂类（工厂模式）
    
    统一管理模型的创建逻辑，支持：
    - 配置驱动的模型创建
    - 字符串名称到模型类的映射
    - 默认参数与用户参数的合并
    """
    
    # 模型默认参数配置
    DEFAULT_CONFIGS: Dict[str, Dict[str, Any]] = {
        'SwinIR': {
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 8,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffle',
            'resi_connection': '1conv',
        },
        'MambaIRv2': {
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'img_range': 1.,
            'embed_dim': 180,
            'depths': [6, 6, 6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6, 6, 6],
            'upsampler': 'pixelshuffle',
        },
        'CWFNet': {
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'img_range': 1.,
            'embed_dim': 180,
            'depths': [6, 6, 6, 6, 6, 6],
            'num_heads': [6, 6, 6, 6, 6, 6],
            'upsampler': 'pixelshuffle',
            'use_wavelet_hint': True,
            'use_cvoca': True,
        },
        'EDSR': {
            'num_in_ch': 3,
            'num_out_ch': 3,
            'num_feat': 64,
            'num_block': 16,
            'upscale': 4,
            'res_scale': 1,
        },
        'ELAN': {
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 8,
            'img_range': 255.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
        },
        'SRFormer': {
            'upscale': 4,
            'in_chans': 3,
            'img_size': 64,
            'window_size': 22,
            'img_range': 1.,
            'depths': [6, 6, 6, 6, 6, 6],
            'embed_dim': 180,
            'num_heads': [6, 6, 6, 6, 6, 6],
            'mlp_ratio': 2,
            'upsampler': 'pixelshuffle',
        },
    }
    
    @classmethod
    def create(cls, config: Dict[str, Any]) -> nn.Module:
        """
        通过配置字典创建模型
        
        Args:
            config: 配置字典，必须包含 'model_name' 或 'name' 键
            
        Returns:
            创建的模型实例
            
        Raises:
            ValueError: 如果配置中缺少模型名称
            KeyError: 如果模型未注册
        """
        model_name = config.get('model_name') or config.get('name')
        if model_name is None:
            raise ValueError("Config must contain 'model_name' or 'name' key")
        
        # 复制配置，避免修改原始字典
        model_config = config.copy()
        model_config.pop('model_name', None)
        model_config.pop('name', None)
        
        return cls.create_by_name(model_name, **model_config)
    
    @classmethod
    def create_by_name(cls, name: str, **kwargs) -> nn.Module:
        """
        通过模型名称创建模型
        
        Args:
            name: 模型名称（需已注册到 MODEL_REGISTRY）
            **kwargs: 模型参数，会覆盖默认参数
            
        Returns:
            创建的模型实例
        """
        # 获取默认配置
        default_config = cls.DEFAULT_CONFIGS.get(name, {})
        
        # 合并默认配置与用户配置
        final_config = {**default_config, **kwargs}
        
        # 从注册表获取模型类
        model_cls = MODEL_REGISTRY.get(name)
        
        return model_cls(**final_config)
    
    @classmethod
    def register_default_config(cls, name: str, config: Dict[str, Any]) -> None:
        """
        注册模型的默认配置
        
        Args:
            name: 模型名称
            config: 默认配置字典
        """
        cls.DEFAULT_CONFIGS[name] = config
    
    @classmethod
    def list_available_models(cls) -> list:
        """返回所有可用的模型名称"""
        return MODEL_REGISTRY.list_registered()
    
    @classmethod
    def get_default_config(cls, name: str) -> Dict[str, Any]:
        """
        获取模型的默认配置
        
        Args:
            name: 模型名称
            
        Returns:
            默认配置字典的副本
        """
        return cls.DEFAULT_CONFIGS.get(name, {}).copy()
