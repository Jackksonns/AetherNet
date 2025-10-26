
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Import the original MambaIRv2 backbone
    from mambairv2_arch import MambaIRv2
except Exception as e:
    raise ImportError("Please ensure mambairv2_arch.py is importable (same folder or PYTHONPATH). Original error: %s" % e)


# ----------------------------- Wavelet (DWT) module -----------------------------
class DWT2D_Haar(nn.Module):
    """
    Fixed 2x2 Haar DWT implemented by grouped conv, applied per-channel.
    Input: (B, C, H, W)  ->  Output: (B, 4*C, H/2, W/2) ordered as [LL, LH, HL, HH]
    """
    def __init__(self):
        super().__init__()
        k = torch.tensor([
            [[0.5,  0.5],
             [0.5,  0.5]],   # LL
            [[0.5, -0.5],
             [0.5, -0.5]],   # LH
            [[0.5,  0.5],
             [-0.5, -0.5]],  # HL
            [[0.5, -0.5],
             [-0.5,  0.5]],  # HH
        ], dtype=torch.float32)  # (4, 2, 2)
        self.register_buffer('k2x2', k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        weight = self.k2x2.view(4, 1, 2, 2).repeat(C, 1, 1, 1)  # (4C,1,2,2)
        y = F.conv2d(x, weight, stride=2, padding=0, groups=C)
        return y


def _split_bands(y: torch.Tensor, C: int):
    """ Split DWT output channels into (LL,LH,HL,HH), each (B,C,H/2,W/2). """
    LL = y[:, 0*C:1*C, ...]
    LH = y[:, 1*C:2*C, ...]
    HL = y[:, 2*C:3*C, ...]
    HH = y[:, 3*C:4*C, ...]
    return LL, LH, HL, HH


# ----------------------------- Hybrid: Wavelet-front + MambaIRv2 -----------------------------
class MambaIRv2Wavelet(MambaIRv2):
    """
    MambaIRv2 with an optional Wavelet-front 'hint' branch:
    - Compute one-level DWT on the input, upsample subbands to original resolution,
      fuse with 1x1 conv to embed_dim, and add to shallow conv features before the backbone.
    - All padding, mask, and reconstruction follow the original MambaIRv2 implementation.
    """
    def __init__(self, *args, use_wavelet_hint: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_wavelet_hint = use_wavelet_hint
        if self.use_wavelet_hint:
            self.dwt = DWT2D_Haar()
            self.wavelet_fuse = nn.Conv2d(self.in_chans if hasattr(self, 'in_chans') else kwargs.get('in_chans', 3) * 4, 
                                          self.embed_dim, kernel_size=1, bias=True)
            # Lightweight gating to avoid over-emphasizing high-frequency noise
            self.wavelet_gate = nn.Sequential(
                nn.Conv2d(self.embed_dim, self.embed_dim, 1), nn.SiLU(),
                nn.Conv2d(self.embed_dim, self.embed_dim, 1)
            )

    @staticmethod
    def _pad_to_even_hw(x: torch.Tensor):
        """ Ensure H and W are even for DWT branch only; return (x_padded, pad_w, pad_h). """
        _, _, h, w = x.size()
        pad_h = h % 2
        pad_w = w % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
        return x, pad_w, pad_h

    def _maybe_wavelet_hint(self, x_normed: torch.Tensor) -> torch.Tensor:
        """
        Compute DWT-hint features from normalized & padded input x_normed (B,C,H,W),
        upsample back to (H,W), fuse to embed_dim and gate.
        """
        if not self.use_wavelet_hint:
            return None
        x_dwt, _, _ = self._pad_to_even_hw(x_normed)
        y = self.dwt(x_dwt)  # (B, 4*C, H/2, W/2)
        C_in = x_normed.shape[1]
        LL, LH, HL, HH = _split_bands(y, C_in)
        up = lambda t: F.interpolate(t, scale_factor=2, mode='nearest')
        wcat = torch.cat([up(LL), up(LH), up(HL), up(HH)], dim=1)  # (B, 4*C, H, W)
        wfeat = self.wavelet_fuse(wcat)  # (B, embed_dim, H, W)
        return self.wavelet_gate(wfeat)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # -- padding to multiples of window_size (follow original MambaIRv2) --
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        mod = self.window_size
        h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        h, w = h_ori + h_pad, w_ori + w_pad
        # symmetric pad by flip like original
        x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]

        # -- normalization to training domain --
        self.mean = self.mean.type_as(x)
        x_normed = (x - self.mean) * self.img_range

        # -- build attention-related params --
        attn_mask = self.calculate_mask([h, w]).to(x.device)
        params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}

        # -- shallow features --
        x_first = self.conv_first(x_normed)  # (B, embed_dim, H, W)

        # -- wavelet hint (optional) --
        if self.use_wavelet_hint:
            wfeat = self._maybe_wavelet_hint(x_normed)  # (B, embed_dim, H, W)
            x_first = x_first + wfeat

        # -- deep features + reconstruction (mirror original branches) --
        if self.upsampler == 'pixelshuffle':
            feat = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            feat = self.conv_before_upsample(feat)
            out = self.conv_last(self.upsample(feat))
        elif self.upsampler == 'pixelshuffledirect':
            feat = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            out = self.upsample(feat)
        elif self.upsampler == 'nearest+conv':
            feat = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            feat = self.conv_before_upsample(feat)
            # nearest-interp branches follow the original file
            feat = F.leaky_relu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode='nearest')), negative_slope=0.2, inplace=False)
            feat = F.leaky_relu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode='nearest')), negative_slope=0.2, inplace=False)
            out = self.conv_last(F.leaky_relu(self.conv_hr(feat), negative_slope=0.2, inplace=False))
        else:
            # denoising / compression artifact removal (no scale)
            res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
            out = x_normed + self.conv_last(res)

        # -- de-normalize and crop to original spatial size (times scale) --
        out = out / self.img_range + self.mean
        H_out = h_ori * self.upscale if hasattr(self, 'upscale') else h_ori
        W_out = w_ori * self.upscale if hasattr(self, 'upscale') else w_ori
        out = out[..., :H_out, :W_out]
        return out
