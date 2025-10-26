# -----------------------------------------------------------------------------------
# SwinIR: Image Restoration Using Swin Transformer, https://arxiv.org/abs/2108.10257
# Originally Written by Ze Liu, Modified by Jingyun Liang.
# -----------------------------------------------------------------------------------

"""
SwinIR模型实现文件
这个文件实现了基于Swin Transformer的图像超分辨率重建模型。
主要功能包括：
1. 图像超分辨率重建
2. 图像去噪
3. JPEG压缩伪影去除

主要组件：
- Mlp: 多层感知机模块
- WindowAttention: 基于窗口的多头自注意力机制
- SwinTransformerBlock: Swin Transformer的基本构建块
- PatchMerging: 图像块合并层
- BasicLayer: Swin Transformer的基本层
- RSTB: 残差Swin Transformer块
- PatchEmbed: 图像到图像块的嵌入
- PatchUnEmbed: 图像块到图像的逆嵌入
- Upsample: 上采样模块
- SwinIR: 主模型类
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_


class Mlp(nn.Module):
    """
    多层感知机模块
    用于特征的非线性变换，包含两个全连接层和激活函数
    
    参数:
        in_features (int): 输入特征维度
        hidden_features (int, optional): 隐藏层特征维度，默认为输入维度
        out_features (int, optional): 输出特征维度，默认为输入维度
        act_layer (nn.Module): 激活函数，默认为GELU
        drop (float): Dropout比率，默认为0
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)  # 第一个全连接层
        self.act = act_layer()  # 激活函数
        self.fc2 = nn.Linear(hidden_features, out_features)  # 第二个全连接层
        self.drop = nn.Dropout(drop)  # Dropout层

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入张量
            
        返回:
            torch.Tensor: 经过MLP处理后的特征
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    将输入特征图分割成不重叠的窗口
    
    参数:
        x (torch.Tensor): 输入特征图，形状为 (B, H, W, C)
        window_size (int): 窗口大小
        
    返回:
        torch.Tensor: 分割后的窗口，形状为 (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    将窗口特征重组回原始特征图
    
    参数:
        windows (torch.Tensor): 窗口特征，形状为 (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始特征图高度
        W (int): 原始特征图宽度
        
    返回:
        torch.Tensor: 重组后的特征图，形状为 (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    """
    基于窗口的多头自注意力机制
    实现了带有相对位置偏置的窗口注意力机制，支持移位和非移位窗口
    
    参数:
        dim (int): 输入通道数
        window_size (tuple[int]): 窗口的高度和宽度
        num_heads (int): 注意力头的数量
        qkv_bias (bool): 是否在query、key、value中添加可学习的偏置，默认为True
        qk_scale (float): 覆盖默认的qk缩放比例，默认为None
        attn_drop (float): 注意力权重的dropout比率，默认为0.0
        proj_drop (float): 输出的dropout比率，默认为0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # 定义相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 获取窗口内每个token的相对位置索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 将坐标移动到从0开始
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # 定义QKV投影层和输出投影层
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 初始化相对位置偏置表
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (num_windows*B, N, C)
            mask (torch.Tensor): 注意力掩码，形状为 (num_windows, Wh*Ww, Wh*Ww) 或 None
            
        返回:
            torch.Tensor: 经过注意力机制处理后的特征
        """
        B_, N, C = x.shape
        # 计算QKV
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 计算注意力分数
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        # 应用注意力掩码（如果存在）
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        # 计算输出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        """返回模块的额外信息字符串"""
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        """
        计算FLOPs（浮点运算次数）
        
        参数:
            N (int): token长度
            
        返回:
            int: 计算一个窗口的FLOPs
        """
        flops = 0
        # qkv投影
        flops += N * self.dim * 3 * self.dim
        # 注意力计算
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        # 输出投影
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # 最终投影
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    """
    Swin Transformer的基本构建块
    包含窗口注意力机制和MLP，支持移位和非移位窗口
    
    参数:
        dim (int): 输入通道数
        input_resolution (tuple[int]): 输入分辨率
        num_heads (int): 注意力头的数量
        window_size (int): 窗口大小，默认为7
        shift_size (int): SW-MSA的移位大小，默认为0
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率，默认为4.0
        qkv_bias (bool): 是否在query、key、value中添加可学习的偏置，默认为True
        qk_scale (float): 覆盖默认的qk缩放比例，默认为None
        drop (float): Dropout比率，默认为0.0
        attn_drop (float): 注意力权重的dropout比率，默认为0.0
        drop_path (float): 随机深度比率，默认为0.0
        act_layer (nn.Module): 激活函数，默认为nn.GELU
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 如果窗口大小大于输入分辨率，则不进行窗口划分
        if min(self.input_resolution) <= self.window_size:
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size必须在0到window_size之间"

        # 第一个归一化层和注意力层
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        # DropPath层
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        
        # 第二个归一化层和MLP层
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        # 计算注意力掩码
        if self.shift_size > 0:
            attn_mask = self.calculate_mask(self.input_resolution)
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def calculate_mask(self, x_size):
        """
        计算SW-MSA的注意力掩码
        
        参数:
            x_size (tuple[int]): 输入特征图的大小
            
        返回:
            torch.Tensor: 注意力掩码
        """
        H, W = x_size
        img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -self.shift_size),
                    slice(-self.shift_size, None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, x_size):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入特征
            x_size (tuple[int]): 输入特征图的大小
            
        返回:
            torch.Tensor: 经过Swin Transformer块处理后的特征
        """
        H, W = x_size
        B, L, C = x.shape

        # 保存残差连接
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # 循环移位
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # 划分窗口
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        if self.input_resolution == x_size:
            attn_windows = self.attn(x_windows, mask=self.attn_mask)
        else:
            attn_windows = self.attn(x_windows, mask=self.calculate_mask(x_size).to(x.device))

        # 合并窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # 反向循环移位
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        """返回模块的额外信息字符串"""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算一个Swin Transformer块的FLOPs
        """
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    """
    图像块合并层
    将相邻的图像块合并，用于下采样和特征融合
    
    参数:
        input_resolution (tuple[int]): 输入特征的分辨率
        dim (int): 输入通道数
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        # 将4个相邻块的特征合并，并降维到2倍
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入特征，形状为 (B, H*W, C)
            
        返回:
            torch.Tensor: 合并后的特征
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "输入特征大小错误"
        assert H % 2 == 0 and W % 2 == 0, f"输入大小 ({H}*{W}) 必须为偶数"

        x = x.view(B, H, W, C)

        # 提取相邻的2x2图像块
        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        
        # 在通道维度上拼接
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        # 归一化和降维
        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        """返回模块的额外信息字符串"""
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算PatchMerging层的FLOPs
        """
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """
    Swin Transformer的基本层
    包含多个Swin Transformer块和一个可选的PatchMerging层
    
    参数:
        dim (int): 输入通道数
        input_resolution (tuple[int]): 输入分辨率
        depth (int): Transformer块的数量
        num_heads (int): 注意力头的数量
        window_size (int): 局部窗口大小
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率，默认为4.0
        qkv_bias (bool): 是否在query、key、value中添加可学习的偏置，默认为True
        qk_scale (float): 覆盖默认的qk缩放比例，默认为None
        drop (float): Dropout比率，默认为0.0
        attn_drop (float): 注意力权重的dropout比率，默认为0.0
        drop_path (float): 随机深度比率，默认为0.0
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm
        downsample (nn.Module): 层末尾的下采样层，默认为None
        use_checkpoint (bool): 是否使用checkpointing来节省内存，默认为False
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # 构建Swin Transformer块
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # PatchMerging层
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入特征
            x_size (tuple[int]): 输入特征图的大小
            
        返回:
            torch.Tensor: 经过BasicLayer处理后的特征
        """
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        """返回模块的额外信息字符串"""
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算BasicLayer的FLOPs
        """
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class RSTB(nn.Module):
    """
    残差Swin Transformer块 (Residual Swin Transformer Block)
    结合了Swin Transformer和残差连接，用于特征提取
    
    参数:
        dim (int): 输入通道数
        input_resolution (tuple[int]): 输入分辨率
        depth (int): Transformer块的数量
        num_heads (int): 注意力头的数量
        window_size (int): 局部窗口大小
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率，默认为4.0
        qkv_bias (bool): 是否在query、key、value中添加可学习的偏置，默认为True
        qk_scale (float): 覆盖默认的qk缩放比例，默认为None
        drop (float): Dropout比率，默认为0.0
        attn_drop (float): 注意力权重的dropout比率，默认为0.0
        drop_path (float): 随机深度比率，默认为0.0
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm
        downsample (nn.Module): 层末尾的下采样层，默认为None
        use_checkpoint (bool): 是否使用checkpointing来节省内存，默认为False
        img_size (int): 输入图像大小，默认为224
        patch_size (int): 图像块大小，默认为4
        resi_connection (str): 残差连接前的卷积块类型，可选'1conv'或'3conv'
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False,
                 img_size=224, patch_size=4, resi_connection='1conv'):
        super(RSTB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        # 残差组：包含多个Swin Transformer块
        self.residual_group = BasicLayer(dim=dim,
                                         input_resolution=input_resolution,
                                         depth=depth,
                                         num_heads=num_heads,
                                         window_size=window_size,
                                         mlp_ratio=mlp_ratio,
                                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                                         drop=drop, attn_drop=attn_drop,
                                         drop_path=drop_path,
                                         norm_layer=norm_layer,
                                         downsample=downsample,
                                         use_checkpoint=use_checkpoint)

        # 残差连接前的卷积块
        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 使用3个卷积层来节省参数和内存
            self.conv = nn.Sequential(nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim // 4, 1, 1, 0),
                                      nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                      nn.Conv2d(dim // 4, dim, 3, 1, 1))

        # 图像块嵌入和逆嵌入
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim,
            norm_layer=None)

    def forward(self, x, x_size):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入特征
            x_size (tuple[int]): 输入特征图的大小
            
        返回:
            torch.Tensor: 经过RSTB处理后的特征
        """
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size), x_size))) + x

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算RSTB的FLOPs
        """
        flops = 0
        flops += self.residual_group.flops()
        H, W = self.input_resolution
        flops += H * W * self.dim * self.dim * 9
        flops += self.patch_embed.flops()
        flops += self.patch_unembed.flops()

        return flops


class PatchEmbed(nn.Module):
    """
    图像到图像块的嵌入层
    将输入图像分割成不重叠的图像块，并进行线性投影
    
    参数:
        img_size (int): 输入图像大小，默认为224
        patch_size (int): 图像块大小，默认为4
        in_chans (int): 输入图像的通道数，默认为3
        embed_dim (int): 线性投影的输出通道数，默认为96
        norm_layer (nn.Module): 归一化层，默认为None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        # 可选的归一化层
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入图像
            
        返回:
            torch.Tensor: 嵌入后的图像块特征
        """
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算PatchEmbed层的FLOPs
        """
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    """
    图像块到图像的逆嵌入层
    将图像块特征重组回原始图像格式
    
    参数:
        img_size (int): 输入图像大小，默认为224
        patch_size (int): 图像块大小，默认为4
        in_chans (int): 输入图像的通道数，默认为3
        embed_dim (int): 线性投影的输出通道数，默认为96
        norm_layer (nn.Module): 归一化层，默认为None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入图像块特征
            x_size (tuple[int]): 输出图像的大小
            
        返回:
            torch.Tensor: 重组后的图像特征
        """
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.embed_dim, x_size[0], x_size[1])  # B Ph*Pw C
        return x

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算PatchUnEmbed层的FLOPs
        """
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """
    上采样模块
    使用像素重排（Pixel Shuffle）进行图像上采样
    
    参数:
        scale (int): 上采样比例，支持2^n和3
        num_feat (int): 中间特征的通道数
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'不支持的上采样比例 {scale}。支持的比例: 2^n 和 3')
        super(Upsample, self).__init__(*m)


class UpsampleOneStep(nn.Sequential):
    """
    一步上采样模块
    与Upsample的区别在于它只包含一个卷积层和一个像素重排层
    用于轻量级超分辨率重建以节省参数
    
    参数:
        scale (int): 上采样比例，支持2^n和3
        num_feat (int): 中间特征的通道数
        num_out_ch (int): 输出通道数
        input_resolution (tuple[int]): 输入分辨率
    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算UpsampleOneStep层的FLOPs
        """
        H, W = self.input_resolution
        flops = H * W * self.num_feat * 3 * 9
        return flops


class SwinIR(nn.Module):
    """
    SwinIR模型
    基于Swin Transformer的图像超分辨率重建模型
    
    参数:
        img_size (int | tuple(int)): 输入图像大小，默认为64
        patch_size (int | tuple(int)): 图像块大小，默认为1
        in_chans (int): 输入图像的通道数，默认为3
        embed_dim (int): 图像块嵌入维度，默认为96
        depths (tuple(int)): 每个Swin Transformer层的深度
        num_heads (tuple(int)): 不同层中注意力头的数量
        window_size (int): 窗口大小，默认为7
        mlp_ratio (float): MLP隐藏层维度与嵌入维度的比率，默认为4
        qkv_bias (bool): 是否在query、key、value中添加可学习的偏置，默认为True
        qk_scale (float): 覆盖默认的qk缩放比例，默认为None
        drop_rate (float): Dropout比率，默认为0
        attn_drop_rate (float): 注意力权重的dropout比率，默认为0
        drop_path_rate (float): 随机深度比率，默认为0.1
        norm_layer (nn.Module): 归一化层，默认为nn.LayerNorm
        ape (bool): 是否在图像块嵌入中添加绝对位置编码，默认为False
        patch_norm (bool): 是否在图像块嵌入后添加归一化，默认为True
        use_checkpoint (bool): 是否使用checkpointing来节省内存，默认为False
        upscale (int): 上采样比例，图像超分辨率重建为2/3/4/8，去噪和压缩伪影去除为1
        img_range (float): 图像范围，1.0或255.0
        upsampler (str): 重建模块类型，可选'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection (str): 残差连接前的卷积块类型，可选'1conv'/'3conv'
    """

    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 **kwargs):
        super(SwinIR, self).__init__()
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        #####################################################################################################
        ################################### 1, 浅层特征提取 ###################################
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        #####################################################################################################
        ################################### 2, 深层特征提取 ######################################
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        # 将图像分割成不重叠的图像块
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 将图像块合并回图像
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=embed_dim, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # 绝对位置编码
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # 随机深度
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # 随机深度衰减规则

        # 构建残差Swin Transformer块 (RSTB)
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = RSTB(dim=embed_dim,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=depths[i_layer],
                         num_heads=num_heads[i_layer],
                         window_size=window_size,
                         mlp_ratio=self.mlp_ratio,
                         qkv_bias=qkv_bias, qk_scale=qk_scale,
                         drop=drop_rate, attn_drop=attn_drop_rate,
                         drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                         norm_layer=norm_layer,
                         downsample=None,
                         use_checkpoint=use_checkpoint,
                         img_size=img_size,
                         patch_size=patch_size,
                         resi_connection=resi_connection
                         )
            self.layers.append(layer)
        self.norm = norm_layer(self.num_features)

        # 深层特征提取的最后一个卷积层
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # 使用3个卷积层来节省参数和内存
            self.conv_after_body = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        #####################################################################################################
        ################################ 3, 高质量图像重建 ################################
        if self.upsampler == 'pixelshuffle':
            # 经典超分辨率重建
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            # 轻量级超分辨率重建（节省参数）
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # 真实场景超分辨率重建（减少伪影）
            self.conv_before_upsample = nn.Sequential(nn.Conv2d(embed_dim, num_feat, 3, 1, 1),
                                                      nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            if self.upscale == 4:
                self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # 图像去噪和JPEG压缩伪影去除
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        """
        初始化模型权重
        
        参数:
            m (nn.Module): 需要初始化的模块
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        """返回不需要权重衰减的参数名称"""
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        """返回不需要权重衰减的关键字"""
        return {'relative_position_bias_table'}

    def check_image_size(self, x):
        """
        检查并调整输入图像大小，使其能被窗口大小整除
        
        参数:
            x (torch.Tensor): 输入图像
            
        返回:
            torch.Tensor: 调整大小后的图像
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_features(self, x):
        """
        提取深层特征
        
        参数:
            x (torch.Tensor): 输入特征
            
        返回:
            torch.Tensor: 提取的深层特征
        """
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)

        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x

    def forward(self, x):
        """
        前向传播函数
        
        参数:
            x (torch.Tensor): 输入图像
            
        返回:
            torch.Tensor: 重建后的图像
        """
        H, W = x.shape[2:]
        x = self.check_image_size(x)
        
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        if self.upsampler == 'pixelshuffle':
            # 经典超分辨率重建
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.conv_last(self.upsample(x))
        elif self.upsampler == 'pixelshuffledirect':
            # 轻量级超分辨率重建
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.upsample(x)
        elif self.upsampler == 'nearest+conv':
            # 真实场景超分辨率重建
            x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x)) + x
            x = self.conv_before_upsample(x)
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            if self.upscale == 4:
                x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.conv_last(self.lrelu(self.conv_hr(x)))
        else:
            # 图像去噪和JPEG压缩伪影去除
            x_first = self.conv_first(x)
            res = self.conv_after_body(self.forward_features(x_first)) + x_first
            x = x + self.conv_last(res)

        x = x / self.img_range + self.mean

        return x[:, :, :H*self.upscale, :W*self.upscale]

    def flops(self):
        """
        计算FLOPs（浮点运算次数）
        
        返回:
            int: 计算整个模型的FLOPs
        """
        flops = 0
        H, W = self.patches_resolution
        flops += H * W * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += H * W * 3 * self.embed_dim * self.embed_dim
        flops += self.upsample.flops()
        return flops


if __name__ == '__main__':
    upscale = 4
    window_size = 8
    height = (1024 // upscale // window_size + 1) * window_size
    width = (720 // upscale // window_size + 1) * window_size
    model = SwinIR(upscale=2, img_size=(height, width),
                   window_size=window_size, img_range=1., depths=[6, 6, 6, 6],
                   embed_dim=60, num_heads=[6, 6, 6, 6], mlp_ratio=2, upsampler='pixelshuffledirect')
    print(model)
    print(height, width, model.flops() / 1e9)

    x = torch.randn((1, 3, height, width))
    x = model(x)
    print(x.shape)
