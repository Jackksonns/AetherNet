"""
测试文件，用于验证 mambairv2_arch.py 的所有依赖项是否已正确复制
"""

try:
    import math
    import numpy as np
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    print("✓ 成功导入标准库依赖")
except ImportError as e:
    print(f"✗ 导入标准库依赖失败: {e}")

try:
    from basicsr.archs.arch_util import to_2tuple, trunc_normal_
    print("✓ 成功导入 basicsr.archs.arch_util")
except ImportError as e:
    print(f"✗ 导入 basicsr.archs.arch_util 失败: {e}")

try:
    from basicsr.utils.registry import ARCH_REGISTRY
    print("✓ 成功导入 basicsr.utils.registry")
except ImportError as e:
    print(f"✗ 导入 basicsr.utils.registry 失败: {e}")

try:
    # 注意：mamba_ssm 和 einops 是外部依赖，不在本次复制范围内
    from einops import rearrange, repeat
    print("✓ 成功导入 einops")
except ImportError as e:
    print(f"✗ 导入 einops 失败: {e}")

try:
    # 注意：mamba_ssm 是外部依赖，不在本次复制范围内
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    print("✓ 成功导入 mamba_ssm")
except ImportError as e:
    print(f"✗ 导入 mamba_ssm 失败: {e}")

try:
    from basicsr.archs.mambairv2_arch import MambaIRv2
    print("✓ 成功导入 MambaIRv2 架构")
    
    # 尝试创建一个简单的模型实例
    model = MambaIRv2(
        upscale=2,
        img_size=64,
        embed_dim=48,
        d_state=8,
        depths=[1, 1, 1, 1],  # 减少层数以节省测试时间
        num_heads=[1, 1, 1, 1],
        window_size=8,
        inner_rank=16,
        num_tokens=32,
        convffn_kernel_size=3,
        img_range=1.,
        mlp_ratio=1.,
        upsampler='pixelshuffledirect')
    print("✓ 成功创建 MambaIRv2 模型实例")
    
except ImportError as e:
    print(f"✗ 导入或创建 MambaIRv2 模型失败: {e}")
except Exception as e:
    print(f"✗ 创建 MambaIRv2 模型时发生错误: {e}")

print("\n注意：mamba_ssm 和 einops 是外部依赖库，需要通过 pip 安装:")
print("pip install mamba_ssm==1.0.1")
print("pip install einops")