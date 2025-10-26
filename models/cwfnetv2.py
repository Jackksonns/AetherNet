import torch
import torch.nn as nn
import torch.nn.functional as F

# --- import your existing modules ---
from .mambairv2_standalone.basicsr.archs import MambaIRv2
from .network_swinirdwt import RSTB, DWT2D_Haar  # reuse SwinIR's RSTB & DWT
from .covca_v2 import CVOCA_V2  # Import CVOCA V2 module
from .adaptive_covca_v2 import AdaptiveCVOCA_V2  # Import AdaptiveCVOCA V2 module

class CWFNetV2(MambaIRv2):
    """
    Hybrid SR network V2:
      - Wavelet-front hint (one-level DWT)
      - MambaIRv2 backbone
      - Small SwinIR RSTB tail polish
      - Unified gated fusion strategy
      - Improved complex representation for CVOCA

    The goal: keep Mamba's efficiency while using Swin's precision at the end,
    with better feature fusion and complex representation.
    """

    def __init__(self,
                 *args,
                 use_wavelet_hint: bool = True,
                 wavelet_gate: bool = True,
                 # CVOCA parameters
                 use_cvoca: bool = False,
                 cvoca_channels: int = 4,
                 # Adaptive CVOCA parameters
                 use_adaptive_cvoca: bool = False,
                 cvoca_use_denoising: bool = True,
                 cvoca_use_multiscale: bool = True,
                 # Fusion parameters
                 use_unified_fusion: bool = True,
                 # RSTB tail
                 num_tail_rstb: int = 1,
                 tail_depth: int = 2,
                 tail_num_heads: int = None,
                 tail_window_size: int = None,
                 **kwargs):
        super().__init__(*args, **kwargs)

        # ---------- Wavelet hint front ----------
        self.use_wavelet_hint = use_wavelet_hint
        if self.use_wavelet_hint:
            in_chans = kwargs.get('in_chans', 3)
            self.dwt = DWT2D_Haar()
            self.wavelet_fuse = nn.Conv2d(in_chans * 4, self.embed_dim, 1, bias=True)
            if wavelet_gate:
                self.wavelet_gate = nn.Sequential(
                    nn.Conv2d(self.embed_dim, self.embed_dim, 1), nn.SiLU(),
                    nn.Conv2d(self.embed_dim, self.embed_dim, 1)
                )
            else:
                self.wavelet_gate = nn.Identity()

        # ---------- CVOCA front ----------
        self.use_cvoca = use_cvoca
        self.use_adaptive_cvoca = use_adaptive_cvoca
        if self.use_cvoca:
            in_chans = kwargs.get('in_chans', 3)
            if self.use_adaptive_cvoca:
                # 使用适应低质量图像的CVOCA模块
                self.adaptive_cvoca = AdaptiveCVOCA_V2(
                    in_channels=1, 
                    out_channels=cvoca_channels, 
                    kernel_size=3,
                    use_denoising=cvoca_use_denoising,
                    use_multiscale=cvoca_use_multiscale
                )
                # 调整CVOCA输出通道数以匹配embed_dim
                self.cvoca_fuse = nn.Conv2d(cvoca_channels * 2, self.embed_dim, 1, bias=True)
            else:
                # CVOCA模块用于提取相位敏感的高频特征
                self.cvoca = CVOCA_V2(in_channels=1, out_channels=cvoca_channels, kernel_size=3)
                # 调整CVOCA输出通道数以匹配embed_dim
                self.cvoca_fuse = nn.Conv2d(cvoca_channels * 2, self.embed_dim, 1, bias=True)
            
            # 门控机制用于自适应融合
            self.cvoca_gate = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim, 1), nn.SiLU(),
                nn.Conv2d(self.embed_dim, self.embed_dim, 1)
            )

        # ---------- Unified fusion gate ----------
        self.use_unified_fusion = use_unified_fusion
        if self.use_unified_fusion and self.use_wavelet_hint and self.use_cvoca:
            # 统一的注意力门控来融合wavelet和cvoca特征
            self.fusion_gate = nn.Sequential(
                nn.Conv2d(self.embed_dim * 2, self.embed_dim // 2, 1),
                nn.SiLU(),
                nn.Conv2d(self.embed_dim // 2, self.embed_dim * 2, 1),
                nn.Sigmoid()
            )

        # ---------- SwinIR RSTB tail ----------
        self.num_tail_rstb = max(0, int(num_tail_rstb))
        self.tail_window_size = tail_window_size or self.window_size
        heads_seq = kwargs.get('num_heads', (4, 4, 4, 4))
        self.tail_num_heads = int(tail_num_heads) if tail_num_heads else int(heads_seq[0])
        self.tail_depth = int(tail_depth)

        self.tail_blocks = nn.ModuleList()
        for _ in range(self.num_tail_rstb):
            self.tail_blocks.append(
                RSTB(
                    dim=self.embed_dim,
                    input_resolution=(self.patches_resolution[0], self.patches_resolution[1]),
                    depth=self.tail_depth,
                    num_heads=self.tail_num_heads,
                    window_size=self.tail_window_size,
                    mlp_ratio=self.mlp_ratio,
                    qkv_bias=True,
                    drop=0.,
                    attn_drop=0.,
                    drop_path=0.,
                    norm_layer=nn.LayerNorm,
                    resi_connection='1conv'
                )
            )

    # ---------- helper: even pad for DWT ----------
    @staticmethod
    def _pad_to_even_hw(x: torch.Tensor):
        _, _, h, w = x.size()
        pad_h, pad_w = h % 2, w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, pad_w, pad_h

    def _wavelet_hint(self, x_normed: torch.Tensor) -> torch.Tensor:
        if not self.use_wavelet_hint:
            return None
        x_dwt, _, _ = self._pad_to_even_hw(x_normed)
        y = self.dwt(x_dwt)  # (B, 4*C, H/2, W/2)
        up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')
        B, C4, H2, W2 = y.shape
        C = C4 // 4
        LL = y[:, 0*C:1*C]; LH = y[:, 1*C:2*C]
        HL = y[:, 2*C:3*C]; HH = y[:, 3*C:4*C]
        wcat = torch.cat([up(LL), up(LH), up(HL), up(HH)], 1)
        fused = self.wavelet_fuse(wcat)
        return self.wavelet_gate(fused)

    def _cvoca_hint(self, x_normed: torch.Tensor) -> torch.Tensor:
        """
        使用CVOCA提取相位敏感的高频特征
        """
        if not self.use_cvoca:
            return None
            
        if self.use_adaptive_cvoca:
            # 使用改进的复数构建
            x_real, x_imag = self.adaptive_cvoca.build_complex_representation(x_normed)
            
            # CVOCA处理
            cvoca_real, cvoca_imag = self.adaptive_cvoca(x_real, x_imag)
        else:
            # 将输入图像转换为复数表示
            # 实部 = 输入图像的亮度通道 (使用平均值获取灰度图)
            # 虚部 = 图像的梯度特征
            if x_normed.shape[1] == 3:
                x_real = torch.mean(x_normed, dim=1, keepdim=True)  # RGB转灰度作为实部
            else:
                x_real = x_normed  # 已经是单通道
                
            # 使用Sobel算子计算梯度作为虚部
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=x_normed.device)
            sobel_y = sobel_x.T
            sobel_x = sobel_x.view(1, 1, 3, 3)
            sobel_y = sobel_y.view(1, 1, 3, 3)
            
            grad_x = F.conv2d(x_real, sobel_x, padding=1)
            grad_y = F.conv2d(x_real, sobel_y, padding=1)
            x_imag = torch.sqrt(grad_x**2 + grad_y**2)  # 梯度幅度作为虚部
            
            # CVOCA前向传播
            cvoca_real, cvoca_imag = self.cvoca(x_real, x_imag)
        
        # 拼接实部和虚部特征并调整通道数
        cvoca_features = torch.cat([cvoca_real, cvoca_imag], dim=1)
        fused = self.cvoca_fuse(cvoca_features)
        
        # 应用门控机制
        return self.cvoca_gate(fused)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ---- MambaIRv2 padding logic ----
        h_ori, w_ori = x.size()[-2:]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1)//mod)*mod - h_ori
        w_pad = ((w_ori + mod - 1)//mod)*mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad

        # symmetric pad (flip)
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        # normalization
        self.mean = self.mean.type_as(x)
        x_normed = (x - self.mean) * self.img_range

        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # shallow feature extraction
        feat0 = self.conv_first(x_normed)
        
        # 双路径特征提取：Wavelet和CVOCA
        wavelet_feat = self._wavelet_hint(x_normed) if self.use_wavelet_hint else None
        cvoca_feat = self._cvoca_hint(x_normed) if self.use_cvoca else None
        
        # 特征融合
        if self.use_wavelet_hint and self.use_cvoca and self.use_unified_fusion:
            # 使用统一的注意力门控来融合
            fusion_input = torch.cat([wavelet_feat, cvoca_feat], dim=1)
            fusion_weights = self.fusion_gate(fusion_input)  # 新增的融合门控
            feat0 = feat0 + fusion_weights[:, :self.embed_dim] * wavelet_feat + \
                            fusion_weights[:, self.embed_dim:] * cvoca_feat
        else:
            # 原始的独立添加方式
            if self.use_wavelet_hint:
                feat0 = feat0 + wavelet_feat
                
            if self.use_cvoca:
                feat0 = feat0 + cvoca_feat

        # MambaIRv2 backbone
        body_feat = self.forward_features(feat0, params)

        # ---- SwinIR RSTB tail polish ----
        if self.num_tail_rstb > 0:

            # 先将 body_feat 与 feat0 做残差连接
            body_feat = body_feat + feat0
            
            tokens = self.patch_embed(body_feat)  # (B, HW, C)
            x_size = (h, w)
            for blk in self.tail_blocks:
                tokens = blk(tokens, x_size)
            body_feat = self.patch_unembed(tokens, x_size)

        # ---- reconstruction ----
        if self.upsampler == 'pixelshuffle':
            feat = self.conv_after_body(body_feat) + feat0
            feat = self.conv_before_upsample(feat)
            out = self.conv_last(self.upsample(feat))
        elif self.upsampler == 'pixelshuffledirect':
            feat = self.conv_after_body(body_feat) + feat0
            out = self.upsample(feat)
        elif self.upsampler == 'nearest+conv':
            feat = self.conv_after_body(body_feat) + feat0
            feat = self.conv_before_upsample(feat)
            feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')), 0.2, False)
            feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')), 0.2, False)
            out = self.conv_last(F.leaky_relu(self.conv_hr(feat), 0.2, False))
        else:
            # denoising / artifact reduction
            res = self.conv_after_body(body_feat) + feat0
            out = x_normed + self.conv_last(res)

        # ---- de-normalize & crop ----
        out = out / self.img_range + self.mean
        H_out = h_ori * getattr(self, 'upscale', 1)
        W_out = w_ori * getattr(self, 'upscale', 1)
        return out[..., :H_out, :W_out]