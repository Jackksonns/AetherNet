# -*- coding: utf-8 -*-
"""
AetherNet Core Module
=====================
设计模式基础设施，包含：
- Registry: 全局注册表（单例模式）
- ModelFactory: 模型工厂（工厂模式）
- DegradationStrategy: 退化策略（策略模式）
- BaseTrainer / BaseTester: 训练/测试模板（模板方法模式）
"""

from .registry import Registry, MODEL_REGISTRY, DATASET_REGISTRY, LOSS_REGISTRY, METRIC_REGISTRY
from .factory import ModelFactory
from .strategy import DegradationStrategy, ClassicalSRDegradation, ColorDenoiseDegradation, JPEGCARDegradation
from .trainer import BaseTrainer
from .tester import BaseTester

__all__ = [
    'Registry',
    'MODEL_REGISTRY',
    'DATASET_REGISTRY', 
    'LOSS_REGISTRY',
    'METRIC_REGISTRY',
    'ModelFactory',
    'DegradationStrategy',
    'ClassicalSRDegradation',
    'ColorDenoiseDegradation',
    'JPEGCARDegradation',
    'BaseTrainer',
    'BaseTester',
]
