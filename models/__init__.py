# -*- coding: utf-8 -*-
"""
Models Package
==============
图像恢复模型集合

包含的模型：
- SwinIR: 基于Swin Transformer的图像恢复
- MambaIRv2: 基于Mamba的图像恢复
- CWFNet: 混合小波-Mamba-Swin网络
- EDSR: 增强深度残差超分辨率
- ELAN: 高效长距离注意力网络
- SRFormer: 超分辨率Transformer
"""

from core.registry import MODEL_REGISTRY

# 延迟导入以避免循环依赖
def _register_models():
    """注册所有模型到全局注册表"""
    from .network_swinir import SwinIR
    from .cwfnet import CWFNet
    from .edsr import EDSR
    from .elan import ELAN
    from .srformer import SRFormer
    
    # 注册模型
    if 'SwinIR' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register_module('SwinIR', SwinIR)
    if 'CWFNet' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register_module('CWFNet', CWFNet)
    if 'EDSR' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register_module('EDSR', EDSR)
    if 'ELAN' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register_module('ELAN', ELAN)
    if 'SRFormer' not in MODEL_REGISTRY:
        MODEL_REGISTRY.register_module('SRFormer', SRFormer)


# 尝试注册模型（可能因依赖缺失而失败）
try:
    _register_models()
except ImportError as e:
    import warnings
    warnings.warn(f"Some models could not be registered: {e}")

__all__ = [
    'SwinIR',
    'CWFNet', 
    'EDSR',
    'ELAN',
    'SRFormer',
]
