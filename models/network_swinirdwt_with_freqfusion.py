# -----------------------------------------------------------------------------------
# SwinIR + DWT(Haar) + FreqFusion —— 集成版
# 目标：在保持 SwinIR 预训练稳态的前提下，融合 DWT 频带提示与 FreqFusion 的可学习频域融合
# 关键设计：
#   1) WaveletHint：对输入做一次 2×2 Haar DWT，四个子带经 1×1 映射到 embed_dim 后，
#      以门控（零初始化）方式注入到浅层特征，稳定启动；
#   2) FreqFusion：保留轻量两支路（高/低）+ 融合 + 门控（零初始化），
#      融合层的最后一层权重置零，初态≈恒等映射；
#   3) 通道对齐：所有 1×1/融合层均以 embed_dim 或 2*compressed_channels 对齐，避免维度不一致；
#   4) 与原 SwinIR 上采样头/无上采样任务完全兼容。
# -----------------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath, to_2tuple, trunc_normal_

# 直接复用你项目中已验证过的 RSTB，避免重复实现引入误差
# from models.network_swinir import RSTB
from .network_swinir import RSTB

# ============================== DWT（Haar） ==============================
class DWT2D_Haar(nn.Module):
    def __init__(self):
        super().__init__()
        k = torch.tensor([
            [[0.5,  0.5],
             [0.5,  0.5]],  # LL
            [[0.5, -0.5],
             [0.5, -0.5]],  # LH
            [[0.5,  0.5],
             [-0.5, -0.5]],  # HL
            [[0.5, -0.5],
             [-0.5,  0.5]],  # HH
        ], dtype=torch.float32)
        self.register_buffer('k2x2', k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        weight = self.k2x2.view(4, 1, 2, 2).repeat(C, 1, 1, 1)  # (4C,1,2,2)
        return F.conv2d(x, weight, stride=2, padding=0, groups=C)


def _split_bands(y: torch.Tensor, C: int):
    LL = y[:, 0*C:1*C, ...]
    LH = y[:, 1*C:2*C, ...]
    HL = y[:, 2*C:3*C, ...]
    HH = y[:, 3*C:4*C, ...]
    return LL, LH, HL, HH


# ============================== Hamming ==============================
def hamming_window_2d(size: int, device=None, dtype=None):
    if size <= 1:
        w = torch.ones((size, size), dtype=dtype, device=device)
        return w / w.sum().clamp_min(1e-6)
    n = torch.arange(size, device=device, dtype=dtype)
    w1d = 0.54 - 0.46 * torch.cos(2 * math.pi * n / (size - 1))
    w2d = torch.outer(w1d, w1d)
    w2d = w2d / w2d.sum().clamp_min(1e-6)
    return w2d


# ============================== CARAFE (scale=1 here) ==============================
def unfold_2d(x, kernel):
    B, C, H, W = x.shape
    pad = kernel // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    patches = F.unfold(x_pad, kernel_size=kernel, dilation=1, padding=0, stride=1)  # (B, C*K*K, H*W)
    patches = patches.view(B, C, kernel*kernel, H, W)
    return patches


def carafe(x, kernel_weights, kernel_size, group=1, scale=1):
    """
    x:             (B, C, H, W)
    kernel_weights:(B, g, K*K, H, W)
    """
    B, C, H, W = x.shape
    g = max(1, group)
    assert C % g == 0, f"C({C}) must be divisible by group({g})"
    Cg = C // g
    K2 = kernel_size * kernel_size

    patches = unfold_2d(x, kernel_size)     # (B, C, K2, H, W)
    w = kernel_weights.view(B, g, K2, H, W).unsqueeze(2).expand(B, g, Cg, K2, H, W)
    patches = patches.view(B, g, Cg, K2, H, W)
    out = (w * patches).sum(dim=3).reshape(B, C, H, W)

    if scale != 1:
        out = F.interpolate(out, scale_factor=scale, mode='nearest')
    return out


# ============================== FreqFusion Head ==============================
class FreqFusion(nn.Module):
    def __init__(self, hr_channels, lr_channels, scale_factor=1,
                 lowpass_kernel=5, highpass_kernel=3, up_group=1,
                 encoder_kernel=3, encoder_dilation=1,
                 compressed_channels=64, align_corners=False,
                 upsample_mode='nearest',
                 feature_resample=False, comp_feat_upsample=False,
                 use_high_pass=True, use_low_pass=True,
                 hr_residual=True, semi_conv=False, hamming_window=True,
                 feature_resample_norm=True):
        super().__init__()
        self.lowpass_kernel = int(lowpass_kernel)
        self.highpass_kernel = int(highpass_kernel)
        self.up_group = int(up_group)
        self.scale_factor = int(scale_factor)
        self.align_corners = align_corners
        self.feature_resample = feature_resample
        self.comp_feat_upsample = comp_feat_upsample
        self.use_high_pass = use_high_pass
        self.use_low_pass = use_low_pass
        self.hr_residual = hr_residual
        self.semi_conv = semi_conv
        self.hamming_window = hamming_window
        self.feature_resample_norm = feature_resample_norm

        # 压缩到中间通道
        ch_mid = int(compressed_channels)
        self.hr_channel_compressor = nn.Conv2d(hr_channels, ch_mid, 1)
        self.lr_channel_compressor = nn.Conv2d(lr_channels, ch_mid, 1)

        # 让 encoder 直接预测每像素的分组核权重：Cmask = g * K^2
        self.k2_lp = self.up_group * (self.lowpass_kernel  * self.lowpass_kernel)
        self.k2_hp = self.up_group * (self.highpass_kernel * self.highpass_kernel)

        self.content_encoder  = nn.Conv2d(ch_mid, self.k2_lp,
                                          kernel_size=encoder_kernel,
                                          padding=encoder_dilation, dilation=encoder_dilation)
        self.content_encoder2 = nn.Conv2d(ch_mid, self.k2_hp,
                                          kernel_size=encoder_kernel,
                                          padding=encoder_dilation, dilation=encoder_dilation)

        # Hamming windows
        size_lp = max(1, self.lowpass_kernel)
        size_hp = max(1, self.highpass_kernel)
        self.register_buffer('hamming_lowpass',  hamming_window_2d(size_lp, dtype=torch.float32))
        self.register_buffer('hamming_highpass', hamming_window_2d(size_hp, dtype=torch.float32))

        # 记录压缩后的通道，供外部融合层使用
        self.ch_mid = ch_mid

    def kernel_normalizer(self, mask4d, kernel, group, hamming=None):
        """
        mask4d: (B, Cmask, H, W), Cmask = group * K^2
        return : (B, g, K*K, H, W)
        """
        B, C, H, W = mask4d.shape
        g = int(group)
        K = int(kernel); K2 = K * K
        assert C == g * K2, f"mask channels {C} must equal group({g}) * K^2({K2})"
        base = mask4d.view(B, g, K2, H, W).view(B, g, K, K, H, W)
        if hamming is None:
            hamming = hamming_window_2d(K, dtype=mask4d.dtype, device=mask4d.device)
        h = hamming.view(1, 1, K, K, 1, 1)
        base = base * h
        s = base.sum(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        base = base / s
        return base.view(B, g, K2, H, W)

    def forward(self, hr_feat, lr_feat):
        B, C, H, W = hr_feat.shape
        hr_c = self.hr_channel_compressor(hr_feat)   # (B, ch_mid, H, W)
        lr_c = self.lr_channel_compressor(lr_feat)   # (B, ch_mid, H, W)

        m_lp = self.content_encoder(hr_c)            # (B, g*K_lp^2, H, W)
        m_hp = self.content_encoder2(lr_c)           # (B, g*K_hp^2, H, W)

        kh = self.kernel_normalizer(m_hp, self.highpass_kernel, self.up_group, hamming=self.hamming_highpass)
        kl = self.kernel_normalizer(m_lp, self.lowpass_kernel,  self.up_group, hamming=self.hamming_lowpass)

        ch = carafe(hr_c, kh, self.highpass_kernel, group=self.up_group, scale=1)
        cl = carafe(lr_c, kl, self.lowpass_kernel,  group=self.up_group, scale=1)

        if self.hr_residual:
            ch = ch + hr_c

        fuse = torch.cat([ch, cl], dim=1)            # (B, 2*ch_mid, H, W)
        return fuse, ch, cl


# ============================== Upsample（同 SwinIR） ==============================
class Upsample(nn.Sequential):
    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:
            for _ in range(int(math.log(scale, 2))):
                m += [nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1), nn.PixelShuffle(2)]
        elif scale == 3:
            m += [nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1), nn.PixelShuffle(3)]
        else:
            raise ValueError(f'Unsupported scale: {scale}.')
        super().__init__(*m)


class UpsampleOneStep(nn.Module):
    def __init__(self, scale, num_feat, num_out_ch, input_resolution):
        super().__init__()
        self.conv = nn.Conv2d(num_feat, (scale * scale) * num_out_ch, 3, 1, 1)
        self.pixshuffle = nn.PixelShuffle(scale)
    def forward(self, x):
        return self.pixshuffle(self.conv(x))


# ============================== 主模型 ==============================
class SwinIR(nn.Module):
    def __init__(self, img_size=64, patch_size=1, in_chans=3,
                 embed_dim=96, depths=[6, 6, 6, 6], num_heads=[6, 6, 6, 6],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, upscale=2, img_range=1., upsampler='', resi_connection='1conv',
                 use_wavelet_hint=True,           # 默认开启：融合 DWT 优势
                 use_freqfusion_head=True,        # 默认开启：融合 FreqFusion 优势
                 wavelet_inject_at='shallow',     # 'shallow' or 'after_body'
                 **kwargs):
        super().__init__()
        num_in_ch = in_chans; num_out_ch = in_chans; num_feat = 64
        self.img_range = img_range
        if in_chans == 3:
            self.register_buffer('mean', torch.Tensor((0.4488, 0.4371, 0.4040)).view(1, 3, 1, 1))
        else:
            self.register_buffer('mean', torch.zeros(1, 1, 1, 1))
        self.upscale = upscale
        self.upsampler = upsampler
        self.window_size = window_size

        # === Wavelet hint ===
        self.use_wavelet_hint = use_wavelet_hint
        self.wavelet_inject_at = wavelet_inject_at
        if self.use_wavelet_hint:
            self.dwt = DWT2D_Haar()
            # 将四个子带(4*in_chans)映射到 embed_dim，与 conv_first 输出维度一致
            self.wavelet_fuse = nn.Conv2d(4*in_chans, embed_dim, 1, 1, 0)
            self.wavelet_gate = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim, 3, 1, 1), nn.SiLU(), nn.Conv2d(embed_dim, embed_dim, 1)
            )
            # 零起步门控：稳定训练前期
            self.wav_scale = nn.Parameter(torch.zeros(1))

        # === FreqFusion head ===
        self.use_freqfusion_head = use_freqfusion_head
        # 关键：记录“压缩通道数”，后续 ff_head_fuse 用 2*ch_mid 作为输入通道
        self.ff_ch_mid = min(64, embed_dim)
        if self.use_freqfusion_head:
            self.ff_head = FreqFusion(
                hr_channels=embed_dim, lr_channels=embed_dim,
                scale_factor=1, lowpass_kernel=5, highpass_kernel=3,
                up_group=1, encoder_kernel=3, encoder_dilation=1,
                compressed_channels=self.ff_ch_mid,
                align_corners=False, upsample_mode='nearest',
                feature_resample=False, comp_feat_upsample=False,
                use_high_pass=True, use_low_pass=True,
                hr_residual=True, semi_conv=False, hamming_window=True,
                feature_resample_norm=True)
            # 融合层输入通道= 2*compressed_channels（HF 与 Base cat）
            self.ff_head_fuse = nn.Sequential(
                nn.Conv2d(self.ff_ch_mid * 2, embed_dim, 1), nn.SiLU(),
                nn.Conv2d(embed_dim, embed_dim, 1)
            )
            # —— 门控，零起步，初态≈纯 SwinIR —— #
            self.ff_scale = nn.Parameter(torch.zeros(1))
            # —— 把融合头最后一层置零，进一步保证初态=恒等 —— #
            with torch.no_grad():
                last = self.ff_head_fuse[-1]
                if isinstance(last, nn.Conv2d):
                    last.weight.zero_()
                    if last.bias is not None:
                        last.bias.zero_()

        # shallow feature
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # deep feature：堆叠 RSTB
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio

        self.patch_embed = nn.Identity()
        self.layers = nn.ModuleList()
        patches_resolution = (img_size // patch_size, img_size // patch_size)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for i_layer in range(self.num_layers):
            layer = RSTB(
                dim=embed_dim,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer], num_heads=num_heads[i_layer],
                window_size=window_size, mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                norm_layer=nn.LayerNorm,
                use_checkpoint=use_checkpoint,
                img_size=img_size, patch_size=patch_size, resi_connection='1conv'
            )
            self.layers.append(layer)

        self.norm = nn.LayerNorm(self.num_features)
        self.conv_after_body = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )

        # reconstruction head（保持与 SwinIR 一致，确保 x(scale) 输出）
        if self.upsampler == 'pixelshuffle':
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True)
            )
            self.upsample = Upsample(self.upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.upsample = UpsampleOneStep(self.upscale, embed_dim, num_out_ch, (img_size, img_size))
        else:
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def _apply_wavelet_hint(self, x, feat):
        """在 x 上做 DWT，将四个子带映射到 embed_dim，并以门控注入到 feat。
        返回与 feat 同形状的增强后特征。
        """
        B, C, H, W = x.shape
        y = self.dwt(x)                        # (B, 4*C, H/2, W/2)
        # 上采样回 H×W（保持计算量可控与稳定，这里使用最近邻；可改为双线性）
        y = F.interpolate(y, size=(H, W), mode='nearest')
        w = self.wavelet_gate(self.wavelet_fuse(y))  # -> (B, embed_dim, H, W)
        return feat + self.wav_scale * w

    def forward_features(self, x):
        # mean shift 之前的原图/输入张量 x 用于 wavelet 提示
        x0 = self.conv_first(x)  # B,C,H,W
        feat = x0

        # Wavelet 提示注入位置可选：浅层或 body 之后
        if self.use_wavelet_hint and self.wavelet_inject_at == 'shallow':
            feat = self._apply_wavelet_hint(x, feat)

        B, C, H, W = feat.shape
        tokens = feat.flatten(2).transpose(1, 2).contiguous()
        x_size = (H, W)
        for layer in self.layers:
            tokens = layer(tokens, x_size)
        deep = tokens.transpose(1, 2).contiguous().view(B, C, H, W)

        body_out = self.conv_after_body(deep) + x0

        if self.use_wavelet_hint and self.wavelet_inject_at == 'after_body':
            body_out = self._apply_wavelet_hint(x, body_out)

        if self.use_freqfusion_head:
            # 这里使用同一特征作为两支路的基底，保持收敛稳定；
            # 若后续你希望“高频支路接入 DWT 的高频组合”，可在此处替换 hr_feat。
            fuse, ch, cl = self.ff_head(body_out, body_out)   # fuse: (B, 2*ff_ch_mid, H, W)
            refined = self.ff_head_fuse(fuse)                 # -> (B, embed_dim, H, W)
            body_out = body_out + self.ff_scale * refined     # 门控融合

        return body_out

    def forward(self, x):
        self.mean = self.mean.type_as(x)
        if self.img_range == 1.:
            x = x - self.mean
        else:
            x = (x - self.mean) * self.img_range

        feat = self.forward_features(x)

        if self.upsampler == 'pixelshuffle':
            out = self.conv_before_upsample(feat)
            out = self.upsample(out)
            out = self.conv_last(out)
        elif self.upsampler == 'pixelshuffledirect':
            out = self.upsample(feat)
        else:
            out = self.conv_last(feat)

        if self.img_range == 1.:
            out = out + self.mean
        else:
            out = out / self.img_range + self.mean
        return out


if __name__ == "__main__":
    # Smoke test: x4 输出尺寸
    model = SwinIR(upscale=4, img_size=64, in_chans=3,
                   embed_dim=96, depths=[6,6,6,6], num_heads=[6,6,6,6],
                   window_size=8, upsampler='pixelshuffle',
                   use_wavelet_hint=True, use_freqfusion_head=True,
                   wavelet_inject_at='shallow')
    x = torch.randn(1, 3, 16, 16)
    y = model(x)
    print("in:", x.shape, "out:", y.shape)  # 期望 (1,3,64,64)
