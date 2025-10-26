# -*- coding: utf-8 -*-
"""
Fair-from-scratch training for JkNet (no pretrained).
This script mirrors the training pipeline of dwt_main_train_swinir_mambairv2_fair.py,
only replacing the model with JkNet.

Usage (example):
python train_jknet.py --task classical_sr --scale 4 \
  --model_size MambaIRv2-M --training_patch_size 64 --batch_size 4 --epochs 10 \
  --lr 1e-4 --train_dir datasets/DIV2K/train --val_dir datasets/DIV2K/val

"""

import os, math, glob, random
from io import BytesIO
from typing import Tuple
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------------
# Repro & small utils (same as reference)
# -------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _make_img_size_divisible(patch_size, window_size):
    if patch_size % window_size == 0:
        return patch_size
    k = patch_size // window_size
    return max(window_size, k * window_size)

def _make_divisible(n, m):
    if n % m == 0:
        return n
    k = n // m
    return max(m, k * m)

def calculate_psnr(img1, img2, crop_border=0, input_order='HWC'):
    if input_order == 'CHW':
        img1 = img1.transpose(1, 2, 0)
        img2 = img2.transpose(1, 2, 0)
    if crop_border > 0:
        img1 = img1[crop_border:-crop_border, crop_border:-crop_border, :]
        img2 = img2[crop_border:-crop_border, crop_border:-crop_border, :]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 99.0
    return 20 * math.log10(255.0 / math.sqrt(mse))

def _imagelist_from_dir(root, exts=('.png','.jpg','.jpeg','.bmp','.tif','.tiff')):
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f'**/*{e}'), recursive=True)
    files = sorted(files)
    return files

def _center_crop_to_multiple(img, multiple):
    w, h = img.size
    W = (w // multiple) * multiple
    H = (h // multiple) * multiple
    if W == 0 or H == 0:
        W = max(multiple, W); H = max(multiple, H)
        return img.resize((W, H), Image.BICUBIC)
    left = (w - W) // 2; top = (h - H) // 2
    return img.crop((left, top, left + W, top + H))

def _random_crop_pair(hr, scale, lr_patch_target, window):
    """
    Random crop from HR such that LR patch == lr_patch_target (aligned to window multiple);
    then HR = LR * scale.
    """
    lr_patch = _make_img_size_divisible(lr_patch_target, window)
    hr_patch = lr_patch * scale
    w, h = hr.size
    if w < hr_patch or h < hr_patch:
        hr = hr.resize((max(hr_patch, w), max(hr_patch, h)), Image.BICUBIC)
        w, h = hr.size
    left = random.randint(0, w - hr_patch)
    top  = random.randint(0, h - hr_patch)
    hr_patch_img = hr.crop((left, top, left + hr_patch, top + hr_patch))
    lr_patch_img = hr_patch_img.resize((lr_patch, lr_patch), Image.BICUBIC)
    return lr_patch_img, hr_patch_img

def _add_gaussian_noise(pil_img, sigma=15):
    arr = np.array(pil_img).astype(np.float32)
    noise = np.random.normal(0, sigma, arr.shape).astype(np.float32)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def _jpeg_compress(pil_img, quality=40):
    buff = BytesIO()
    pil_img.save(buff, format='JPEG', quality=int(quality))
    buff.seek(0)
    return Image.open(buff).convert(pil_img.mode)

# -------------------------
# Unified Dataset (same behavior as reference)
# -------------------------
class SRDataset(Dataset):
    def __init__(self, root_hr=None, root_lq=None, root_gt=None, task='classical_sr',
                 scale=4, patch=64, noise=0, jpeg=0, split='train', gray=False, window_size=8):
        super().__init__()
        self.task = task
        self.scale = int(scale)
        self.patch = int(patch)
        self.noise = int(noise)
        self.jpeg = int(jpeg)
        self.gray = bool(gray)
        self.split = split
        self.window = int(window_size)

        self.root_hr = root_hr
        self.root_lq = root_lq
        self.root_gt = root_gt

        if split == 'train':
            assert self.root_hr and os.path.isdir(self.root_hr), f"train HR dir invalid: {self.root_hr}"
            self.hr_files = _imagelist_from_dir(self.root_hr)
        else:
            if self.root_lq and self.root_gt and os.path.isdir(self.root_lq) and os.path.isdir(self.root_gt):
                self.lq_files = _imagelist_from_dir(self.root_lq)
                self.gt_files = _imagelist_from_dir(self.root_gt)
                self.pairs = []
                gt_map = {os.path.basename(p): p for p in self.gt_files}
                for p in self.lq_files:
                    name = os.path.basename(p)
                    if name in gt_map:
                        self.pairs.append((p, gt_map[name]))
                if len(self.pairs) == 0:
                    raise RuntimeError("No LQ/GT pairs matched in val folders.")
                self.hr_files = None
            else:
                assert self.root_hr and os.path.isdir(self.root_hr), f"val HR dir invalid: {self.root_hr}"
                self.hr_files = _imagelist_from_dir(self.root_hr)
                self.pairs = None

    def __len__(self):
        if self.split == 'train':
            return len(self.hr_files) * 50  # <-- mimic reference repeat
        else:
            return len(self.pairs) if self.pairs is not None else len(self.hr_files)

    def _load_img(self, path):
        img = Image.open(path).convert('RGB')
        if self.gray:
            img = img.convert('L')
        return img

    def __getitem__(self, idx):
        if self.split == 'train':
            hr = self._load_img(self.hr_files[idx % len(self.hr_files)])
            if self.task == 'classical_sr':
                lr, hr = _random_crop_pair(hr, self.scale, self.patch, self.window)
            elif self.task in ('gray_dn', 'color_dn'):
                w, h = hr.size
                if w < self.patch or h < self.patch:
                    hr = hr.resize((max(self.patch, w), max(self.patch, h)), Image.BICUBIC)
                left = random.randint(0, hr.size[0] - self.patch)
                top = random.randint(0, hr.size[1] - self.patch)
                hr = hr.crop((left, top, left + self.patch, top + self.patch))
                lr = _add_gaussian_noise(hr, sigma=self.noise if self.noise > 0 else 15)
            elif self.task in ('jpeg_car', 'color_jpeg_car'):
                w, h = hr.size
                if w < self.patch or h < self.patch:
                    hr = hr.resize((max(self.patch, w), max(self.patch, h)), Image.BICUBIC)
                left = random.randint(0, hr.size[0] - self.patch)
                top = random.randint(0, hr.size[1] - self.patch)
                hr = hr.crop((left, top, left + self.patch, top + self.patch))
                lr = _jpeg_compress(hr, quality=self.jpeg if self.jpeg > 0 else 40)
            else:
                raise ValueError(f"Unsupported task: {self.task}")
        else:
            if self.pairs is not None:
                lq_path, gt_path = self.pairs[idx]
                lr = self._load_img(lq_path)
                hr = self._load_img(gt_path)
                if self.task == 'classical_sr':
                    W = _make_divisible(lr.size[0], self.window)
                    H = _make_divisible(lr.size[1], self.window)
                    if (W, H) != lr.size:
                        lr = lr.resize((W, H), Image.BICUBIC)
                        hr = hr.resize((W * self.scale, H * self.scale), Image.BICUBIC)
            else:
                hr = self._load_img(self.hr_files[idx])
                if self.task == 'classical_sr':
                    base_multiple = self.scale * self.window
                    hr = _center_crop_to_multiple(hr, base_multiple)
                    w, h = hr.size
                    lr = hr.resize((w // self.scale, h // self.scale), Image.BICUBIC)
                elif self.task in ('gray_dn', 'color_dn'):
                    hr = _center_crop_to_multiple(hr, self.window)
                    lr = _add_gaussian_noise(hr, sigma=self.noise if self.noise > 0 else 15)
                elif self.task in ('jpeg_car', 'color_jpeg_car'):
                    hr = _center_crop_to_multiple(hr, self.window)
                    lr = _jpeg_compress(hr, quality=self.jpeg if self.jpeg > 0 else 40)
                else:
                    raise ValueError(f"Unsupported task: {self.task}")

        # to tensor [0,1]
        if self.gray:
            lr = np.array(lr).astype(np.float32) / 255.0
            hr = np.array(hr).astype(np.float32) / 255.0
            lr = torch.from_numpy(lr).unsqueeze(0)
            hr = torch.from_numpy(hr).unsqueeze(0)
        else:
            lr = torch.from_numpy(np.array(lr).astype(np.float32).transpose(2,0,1)) / 255.0
            hr = torch.from_numpy(np.array(hr).astype(np.float32).transpose(2,0,1)) / 255.0
        return lr, hr

# -------------------------
# Dataloaders (same as reference)
# -------------------------
def build_train_loader(args, window_size):
    gray = (args.task in ('gray_dn', 'jpeg_car'))
    ds = SRDataset(
        root_hr=args.train_dir, task=args.task, scale=args.scale,
        patch=getattr(args, 'training_patch_size', 64),
        noise=getattr(args, 'noise', 0), jpeg=getattr(args, 'jpeg', 0),
        split='train', gray=gray, window_size=window_size
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=4,
                        pin_memory=True, drop_last=True)
    return loader

def build_val_loader(args, window_size):
    gray = (args.task in ('gray_dn', 'jpeg_car'))
    use_pairs = bool(getattr(args, 'folder_val_lq', '')) and bool(getattr(args, 'folder_val_gt', ''))
    if use_pairs:
        ds = SRDataset(root_lq=args.folder_val_lq, root_gt=args.folder_val_gt,
                       task=args.task, scale=args.scale, noise=getattr(args, 'noise', 0),
                       jpeg=getattr(args, 'jpeg', 0), split='val', gray=gray, window_size=window_size)
    else:
        ds = SRDataset(root_hr=args.val_dir, task=args.task, scale=args.scale,
                       noise=getattr(args, 'noise', 0), jpeg=getattr(args, 'jpeg', 0),
                       split='val', gray=gray, window_size=window_size)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    return loader

# -------------------------
# Train / Validate loop (same logic as reference)
# -------------------------
def train(args, define_model_fn, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    set_seed(getattr(args, 'seed', 42))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sanitize by task
    if args.task == 'classical_sr':
        args.noise = 0; args.jpeg = 0
        args.degradation = getattr(args, 'degradation', 'bicubic')
    elif args.task in ('gray_dn', 'color_dn'):
        args.scale = 1; args.degradation = 'gaussian'; args.jpeg = 0
    elif args.task in ('jpeg_car', 'color_jpeg_car'):
        args.scale = 1; args.degradation = 'jpeg'; args.noise = 0

    model, window_size = define_model_fn(args)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.L1Loss()

    train_loader = build_train_loader(args, window_size)
    val_loader = build_val_loader(args, window_size)

    best_psnr = -1.0
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch}/{args.epochs}]", dynamic_ncols=True, leave=False)
        for i, (lq, gt) in enumerate(pbar, 1):
            lq, gt = lq.to(device), gt.to(device)
            pred = model(lq)
            loss = criterion(pred, gt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg_loss=f"{(epoch_loss/i):.4f}")

        avg_loss = epoch_loss / max(1, len(train_loader))
        tqdm.write(f"Epoch [{epoch}/{args.epochs}]  AvgLoss: {avg_loss:.6f}")

        # validate
        if val_loader is not None:
            model.eval()
            psnr_total, seen = 0.0, 0
            with torch.no_grad():
                vbar = tqdm(val_loader, desc="Val", dynamic_ncols=True, leave=False)
                for lq, gt in vbar:
                    lq, gt = lq.to(device), gt.to(device)
                    out = model(lq)
                    out_np = out.squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)
                    gt_np  = gt .squeeze(0).permute(1,2,0).detach().cpu().numpy().clip(0,1)
                    out_np = (out_np * 255.0).round()
                    gt_np  = (gt_np  * 255.0).round()
                    border = args.scale if args.task in ('classical_sr','lightweight_sr','real_sr') else 0
                    psnr = calculate_psnr(out_np, gt_np, crop_border=border, input_order='HWC')
                    psnr_total += psnr; seen += 1
                    vbar.set_postfix(psnr=f"{(psnr_total/max(1,seen)):.2f} dB")
            val_psnr = psnr_total / max(1, seen)
            tqdm.write(f"Validation PSNR: {val_psnr:.2f} dB")

            if val_psnr > best_psnr:
                best_psnr = val_psnr
                no_improve = 0
                save_path = os.path.join(save_dir,
                    f"best_task-{args.task}_scale-{args.scale}_patch-{getattr(args,'training_patch_size',64)}_"
                    f"{args.model_size}_bs-{args.batch_size}_lr-{args.lr}_psnr-{val_psnr:.2f}.pth")
                torch.save(model.state_dict(), save_path)
                tqdm.write(f"[Saved] {save_path}")
            else:
                no_improve += 1
                tqdm.write(f"No improvement for {no_improve} epoch(s).")
                if no_improve >= getattr(args, 'patience', 5):
                    tqdm.write(f"[Early Stop] at epoch {epoch}")
                    break
    return model

# -------------------------
# Model (JkNet)
# -------------------------
from models.jknet import JkNet  # <-- your JkNet model

def define_model(args):
    """
    Map CLI args -> JkNet config.
    We mirror the SwinIR reference: window=7 for jpeg tasks, else 8;
    embed_dim by model_size; heads/depths follow a uniform shallow config.
    """
    jpeg_tasks = {'jpeg_car', 'color_jpeg_car'}
    task = getattr(args, 'task', '')
    window_size = 7 if task in jpeg_tasks else 8
    training_patch = int(getattr(args, 'training_patch_size', 64))
    model_img_size = _make_img_size_divisible(training_patch, window_size)

    # Keep same size presets as SwinIR reference for fairness
    model_configs = {
        'SwinIR-S': dict(embed_dim=60,  upsampler='pixelshuffledirect', resi_connection='1conv'),
        'SwinIR-M': dict(embed_dim=180, upsampler='pixelshuffle',       resi_connection='1conv'),
        'SwinIR-L': dict(embed_dim=240, upsampler='pixelshuffle',       resi_connection='1conv'),
        'MambaIRv2-S': dict(embed_dim=60,  upsampler='pixelshuffledirect', resi_connection='1conv'),
        'MambaIRv2-M': dict(embed_dim=180, upsampler='pixelshuffle',       resi_connection='1conv'),
        'MambaIRv2-L': dict(embed_dim=240, upsampler='pixelshuffle',       resi_connection='1conv'),
    }
    if args.model_size not in model_configs:
        raise ValueError(f"Unknown model_size {args.model_size}")
    cfg = model_configs[args.model_size]

    use_sr_upsampler = (task in ('classical_sr', 'lightweight_sr', 'real_sr')) and (args.scale > 1)
    upsampler = cfg['upsampler'] if use_sr_upsampler else ''

    # Heads/depths mimic the reference (uniform stacks)
    depths = [6]*6
    num_heads = [6]*6
    mlp_ratio = 2

    # For denoise/jpeg tasks, SwinIR 使用不同 img_range；我们保持一致性
    img_range = 1.0 if task in ('classical_sr','lightweight_sr','real_sr') else 255.0

    # JkNet-specific knobs: wavelet hint, FPA, and RSTB tail
    model = JkNet(
        upscale=args.scale,
        in_chans=3 if 'gray' not in task else 1,
        img_size=model_img_size,
        window_size=window_size,
        img_range=img_range,
        depths=depths,
        embed_dim=cfg['embed_dim'],
        num_heads=tuple(num_heads),
        mlp_ratio=mlp_ratio,
        upsampler=upsampler,
        resi_connection=cfg['resi_connection'] if task in ('classical_sr','lightweight_sr') else (
            '3conv' if (task=='real_sr' and getattr(args,'large_model',False)) else '1conv'
        ),
        # --- JkNet extras ---
        use_wavelet_hint=True,
        wavelet_gate=True,
        use_fpa=True,
        num_tail_rstb=1,
        tail_depth=2,
        tail_num_heads=None,      # default -> use first of num_heads
        tail_window_size=None     # default -> window_size
    )

    # sanity check SR upsampler geometry like reference
    if use_sr_upsampler:
        with torch.no_grad():
            # 确保模型在验证前已经在正确的设备上
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
            
            C = 1 if ('gray' in task) else 3
            dummy = torch.randn(1, C, 16, 16)
            # 确保dummy张量在GPU上，如果可用的话
            if torch.cuda.is_available():
                dummy = dummy.cuda()
            out = model(dummy)
            assert out.shape[-2:] == (16*args.scale, 16*args.scale), "SR head must scale correctly"
    return model, window_size

# -------------------------
# CLI
# -------------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Fair-from-scratch: JkNet")
    parser.add_argument("--task", type=str, default="classical_sr",
                        choices=["classical_sr","lightweight_sr","real_sr","gray_dn","color_dn","jpeg_car","color_jpeg_car"])
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--model_size", type=str, default="SwinIR-M", choices=["SwinIR-S","SwinIR-M","SwinIR-L", "MambaIRv2-S", "MambaIRv2-M", "MambaIRv2-L"])
    parser.add_argument("--large_model", action="store_true")

    parser.add_argument("--training_patch_size", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--train_dir", type=str, default="datasets/train/GT", help="Training data directory")
    parser.add_argument("--val_dir", type=str, default="datasets/val", help="Validation data directory")
    parser.add_argument("--folder_val_lq", type=str, default="", help="Paired validation LQ directory")
    parser.add_argument("--folder_val_gt", type=str, default="", help="Paired validation GT directory")
    parser.add_argument("--noise", type=int, default=0, help="Noise level for dn tasks")
    parser.add_argument("--jpeg", type=int, default=0, help="JPEG quality for jpeg tasks")
    parser.add_argument("--degradation", type=str, default="bicubic", help="Degradation type for SR tasks")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--save_dir", type=str, default="model_zoo/JkNet", help="Save directory")
    args = parser.parse_args()
    train(args, define_model, args.save_dir)

if __name__ == "__main__":
    main()