import torch
import torch.nn as nn
import torch.nn.functional as F

# --- import your existing modules ---
from .mambairv2_standalone.basicsr.archs import MambaIRv2
from .network_swinirdwt import RSTB, DWT2D_Haar  # reuse SwinIR's RSTB & DWT

class DwtSwinMambaIRv2(MambaIRv2):
    """
    Hybrid SR network:
      - Wavelet-front hint (one-level DWT)
      - MambaIRv2 backbone
      - Small SwinIR RSTB tail polish

    The goal: keep Mamba's efficiency while using Swin's precision at the end.
    """

    def __init__(self,
                 *args,
                 use_wavelet_hint: bool = True,
                 wavelet_gate: bool = True,
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
        if self.use_wavelet_hint:
            feat0 = feat0 + self._wavelet_hint(x_normed)

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