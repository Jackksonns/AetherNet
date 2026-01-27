# -*- coding: utf-8 -*-
"""
Registry Module - 注册表模式实现
================================
提供全局注册表，用于动态注册和获取模型、数据集、损失函数、评价指标等组件。
支持装饰器语法和字符串名称查找。

Usage:
    >>> from core import MODEL_REGISTRY
    >>> 
    >>> @MODEL_REGISTRY.register()
    >>> class SwinIR(nn.Module):
    >>>     ...
    >>>
    >>> model_cls = MODEL_REGISTRY.get('SwinIR')
    >>> model = model_cls(**config)
"""

from typing import Any, Dict, Optional, Callable, Type


class Registry:
    """
    通用注册表类（单例模式思想）
    
    用于管理和查找各类组件，如模型、数据集、损失函数等。
    
    Attributes:
        _name (str): 注册表名称
        _registry (dict): 存储已注册对象的字典
    """
    
    def __init__(self, name: str):
        """
        初始化注册表
        
        Args:
            name: 注册表名称，用于标识和调试
        """
        self._name = name
        self._registry: Dict[str, Any] = {}
    
    @property
    def name(self) -> str:
        """返回注册表名称"""
        return self._name
    
    def __len__(self) -> int:
        """返回已注册对象数量"""
        return len(self._registry)
    
    def __contains__(self, key: str) -> bool:
        """检查是否包含指定键"""
        return key in self._registry
    
    def __repr__(self) -> str:
        return f"Registry(name={self._name}, items={list(self._registry.keys())})"
    
    def register(self, name: Optional[str] = None) -> Callable:
        """
        装饰器：注册一个类或函数
        
        Args:
            name: 注册名称，默认使用类/函数名
            
        Returns:
            装饰器函数
            
        Example:
            >>> @registry.register()
            >>> class MyModel:
            >>>     pass
            >>>
            >>> @registry.register('custom_name')
            >>> class AnotherModel:
            >>>     pass
        """
        def decorator(obj: Type) -> Type:
            key = name if name is not None else obj.__name__
            if key in self._registry:
                raise KeyError(f"'{key}' is already registered in {self._name}")
            self._registry[key] = obj
            return obj
        return decorator
    
    def register_module(self, name: str, module: Any) -> None:
        """
        直接注册一个模块（非装饰器方式）
        
        Args:
            name: 注册名称
            module: 要注册的对象
        """
        if name in self._registry:
            raise KeyError(f"'{name}' is already registered in {self._name}")
        self._registry[name] = module
    
    def get(self, name: str) -> Any:
        """
        根据名称获取已注册对象
        
        Args:
            name: 注册名称
            
        Returns:
            已注册的对象
            
        Raises:
            KeyError: 如果名称未注册
        """
        if name not in self._registry:
            raise KeyError(f"'{name}' is not registered in {self._name}. "
                          f"Available: {list(self._registry.keys())}")
        return self._registry[name]
    
    def list_registered(self) -> list:
        """返回所有已注册的名称列表"""
        return list(self._registry.keys())


# ==================== 全局注册表实例 ====================

MODEL_REGISTRY = Registry('models')
"""模型注册表：用于注册所有网络模型"""

DATASET_REGISTRY = Registry('datasets')
"""数据集注册表：用于注册数据集类"""

LOSS_REGISTRY = Registry('losses')
"""损失函数注册表：用于注册损失函数"""

METRIC_REGISTRY = Registry('metrics')
"""评价指标注册表：用于注册评价指标函数"""
