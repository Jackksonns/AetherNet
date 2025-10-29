# -*- coding: utf-8 -*-
"""
EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution
基于官方EDSR-PyTorch实现，适配AetherNet项目

Reference:
- Enhanced Deep Residual Networks for Single Image Super-Resolution (CVPRW 2017)
- https://github.com/sanghyun-son/EDSR-PyTorch
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    """默认卷积层"""
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)


class MeanShift(nn.Conv2d):
    """
    RGB均值偏移层
    用于将RGB图像的均值归一化到0
    """
    def __init__(
        self, rgb_range=1.0,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):
        
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class ResBlock(nn.Module):
    """
    EDSR的残差块
    结构：Conv-ReLU-Conv + res_scale
    """
    def __init__(
        self, conv, n_feats, kernel_size,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1):
        
        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(n_feats, n_feats, kernel_size, bias=bias))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Upsampler(nn.Sequential):
    """
    上采样模块
    使用PixelShuffle进行上采样，支持x2, x3, x4等多种尺度
    """
    def __init__(self, conv, scale, n_feats, bn=False, act=False, bias=True):
        
        m = []
        if (scale & (scale - 1)) == 0:    # Is scale = 2^n?
            for _ in range(int(math.log(scale, 2))):
                m.append(conv(n_feats, 4 * n_feats, 3, bias))
                m.append(nn.PixelShuffle(2))
                if bn:
                    m.append(nn.BatchNorm2d(n_feats))
                if act == 'relu':
                    m.append(nn.ReLU(True))
                elif act == 'prelu':
                    m.append(nn.PReLU(n_feats))

        elif scale == 3:
            m.append(conv(n_feats, 9 * n_feats, 3, bias))
            m.append(nn.PixelShuffle(3))
            if bn:
                m.append(nn.BatchNorm2d(n_feats))
            if act == 'relu':
                m.append(nn.ReLU(True))
            elif act == 'prelu':
                m.append(nn.PReLU(n_feats))
        else:
            raise NotImplementedError

        super(Upsampler, self).__init__(*m)


class EDSR(nn.Module):
    """
    EDSR: Enhanced Deep Residual Networks for Single Image Super-Resolution
    
    Args:
        n_resblocks (int): 残差块数量（baseline: 16, standard: 32）
        n_feats (int): 特征通道数（baseline: 64, standard: 256）
        kernel_size (int): 卷积核大小，默认3
        scale (int): 上采样倍数（2, 3, 4等）
        res_scale (float): 残差缩放因子（baseline: 1, standard: 0.1）
        n_colors (int): 输入图像通道数，默认3（RGB）
        rgb_range (float): RGB值范围，默认1.0
    """
    def __init__(
        self, 
        n_resblocks=16, 
        n_feats=64, 
        kernel_size=3,
        scale=4, 
        res_scale=1.0,
        n_colors=3,
        rgb_range=1.0,
        conv=default_conv):
        
        super(EDSR, self).__init__()

        self.scale = scale
        self.n_resblocks = n_resblocks
        self.n_feats = n_feats

        # RGB均值
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = MeanShift(rgb_range, rgb_mean, rgb_std)
        
        # 头部：单个卷积层
        m_head = [conv(n_colors, n_feats, kernel_size)]

        # 主体：多个残差块
        m_body = [
            ResBlock(
                conv, n_feats, kernel_size, 
                act=nn.ReLU(True), res_scale=res_scale
            ) for _ in range(n_resblocks)
        ]
        m_body.append(conv(n_feats, n_feats, kernel_size))

        # 尾部：上采样 + 输出卷积
        m_tail = [
            Upsampler(conv, scale, n_feats, act=False),
            conv(n_feats, n_colors, kernel_size)
        ]

        # 添加均值还原
        self.add_mean = MeanShift(rgb_range, rgb_mean, rgb_std, 1)

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        # 减去均值
        x = self.sub_mean(x)
        
        # 头部特征提取
        x = self.head(x)

        # 主体残差学习（带全局残差连接）
        res = self.body(x)
        res += x

        # 尾部上采样和重建
        x = self.tail(res)
        
        # 加上均值
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=True):
        """自定义加载函数，处理可能的键名不匹配"""
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


def make_edsr_baseline(scale=4, n_colors=3, rgb_range=1.0):
    """创建EDSR-baseline模型"""
    return EDSR(
        n_resblocks=16, 
        n_feats=64, 
        scale=scale, 
        res_scale=1.0,
        n_colors=n_colors,
        rgb_range=rgb_range
    )


def make_edsr(scale=4, n_colors=3, rgb_range=1.0):
    """创建EDSR标准模型"""
    return EDSR(
        n_resblocks=32, 
        n_feats=256, 
        scale=scale, 
        res_scale=0.1,
        n_colors=n_colors,
        rgb_range=rgb_range
    )

