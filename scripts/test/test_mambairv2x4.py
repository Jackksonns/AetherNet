import os
import argparse
import glob
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import time
from pathlib import Path
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Import MambaIRv2 model
from models.mambairv2_standalone.basicsr.archs.mambairv2_arch import MambaIRv2


def _make_img_size_divisible(patch_size, window_size):
    """Make patch size divisible by window size"""
    if patch_size % window_size == 0:
        return patch_size
    return int(np.ceil(patch_size / window_size) * window_size)


def _make_tensor_divisible(img_tensor, window_size):
    """Make image tensor dimensions divisible by window size"""
    h, w = img_tensor.shape[2], img_tensor.shape[3]
    h_pad = (window_size - h % window_size) % window_size
    w_pad = (window_size - w % window_size) % window_size
    img_tensor = torch.nn.functional.pad(img_tensor, (0, w_pad, 0, h_pad), mode='reflect')
    return img_tensor


def define_model(args):
    """Define MambaIRv2 model based on args"""
    # 从项目中导入MambaIRv2模型
    from models.mambairv2_standalone.basicsr.archs.mambairv2_arch import MambaIRv2
    
    # Task-specific configs
    jpeg_tasks = {'jpeg_car', 'color_jpeg_car'}
    task = getattr(args, 'task', 'classical_sr')
    window_size = 7 if task in jpeg_tasks else 16
    training_patch = int(getattr(args, 'training_patch_size', 64))
    model_img_size = _make_img_size_divisible(training_patch, window_size)

    # Model configs (following SwinIR reference for fairness)
    model_configs = {
        'tiny': dict(embed_dim=60,  upsampler='pixelshuffledirect', resi_connection='1conv'),
        'light': dict(embed_dim=180, upsampler='pixelshuffle',       resi_connection='1conv'),
        'medium': dict(embed_dim=240, upsampler='pixelshuffle',       resi_connection='1conv'),
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

    # For denoise/jpeg tasks, SwinIR uses different img_range; we keep consistency
    img_range = 1.0 if task in ('classical_sr','lightweight_sr','real_sr') else 255.0

    # MambaIRv2-specific parameters
    model = MambaIRv2(
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
        # MambaIRv2 specific parameters
        d_state=16,
        inner_rank=32,
        num_tokens=64,
        convffn_kernel_size=5,
        qkv_bias=True,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
    )
    return model, window_size


def _imagelist_from_dir(root):
    """Get list of image files from directory"""
    exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
    files = []
    for e in exts:
        files += glob.glob(os.path.join(root, f'**/*{e}'), recursive=True)
    files = sorted(files)
    return files


class TestDataset(Dataset):
    """Test dataset for loading LQ and GT image pairs"""
    
    def __init__(self, lq_dir=None, gt_dir=None, task='classical_sr', scale=4, gray=False):
        super().__init__()
        self.task = task
        self.scale = int(scale)
        self.gray = bool(gray)
        
        self.lq_dir = lq_dir
        self.gt_dir = gt_dir
        
        # Load LQ and GT file pairs
        if lq_dir and gt_dir and os.path.isdir(lq_dir) and os.path.isdir(gt_dir):
            self.lq_files = _imagelist_from_dir(lq_dir)
            self.gt_files = _imagelist_from_dir(gt_dir)
            self.pairs = []
            gt_map = {os.path.basename(p): p for p in self.gt_files}
            for p in self.lq_files:
                name = os.path.basename(p)
                if name in gt_map:
                    self.pairs.append((p, gt_map[name]))
            print(f"Found {len(self.pairs)} paired test images")
        else:
            raise ValueError("Both LQ and GT directories must be provided for testing")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        lq_path, gt_path = self.pairs[index]
        
        # Load images
        lq = cv2.imread(lq_path, cv2.IMREAD_COLOR)
        gt = cv2.imread(gt_path, cv2.IMREAD_COLOR)
        
        # BGR to RGB
        lq = cv2.cvtColor(lq, cv2.COLOR_BGR2RGB)
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        
        # Convert to tensor and normalize to [0, 1]
        if self.gray:
            lq_tensor = torch.from_numpy(lq.astype(np.float32) / 255.0).unsqueeze(0)
            gt_tensor = torch.from_numpy(gt.astype(np.float32) / 255.0).unsqueeze(0)
        else:
            lq_tensor = torch.from_numpy(lq.astype(np.float32).transpose(2, 0, 1)) / 255.0
            gt_tensor = torch.from_numpy(gt.astype(np.float32).transpose(2, 0, 1)) / 255.0
            
        filename = os.path.basename(lq_path)
        return lq_tensor, gt_tensor, filename


def calculate_psnr(img1, img2, border=0):
    """Calculate PSNR between two images"""
    if img1.ndim == 3:
        img1 = img1.transpose(1, 2, 0)
    if img2.ndim == 3:
        img2 = img2.transpose(1, 2, 0)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Crop border if specified
    if border > 0:
        img1 = img1[border:-border, border:-border, :]
        img2 = img2[border:-border, border:-border, :]
    
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(1.0 / np.sqrt(mse))


def calculate_ssim(img1, img2, border=0):
    """Calculate SSIM between two images"""
    if img1.ndim == 3:
        img1 = img1.transpose(1, 2, 0)
    if img2.ndim == 3:
        img2 = img2.transpose(1, 2, 0)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Crop border if specified
    if border > 0:
        img1 = img1[border:-border, border:-border, :]
        img2 = img2[border:-border, border:-border, :]
    
    return ssim(img1, img2, channel_axis=2, data_range=1.0)


def calculate_rmse(img1, img2, border=0):
    """Calculate RMSE between two images"""
    if img1.ndim == 3:
        img1 = img1.transpose(1, 2, 0)
    if img2.ndim == 3:
        img2 = img2.transpose(1, 2, 0)
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    # Crop border if specified
    if border > 0:
        img1 = img1[border:-border, border:-border, :]
        img2 = img2[border:-border, border:-border, :]
    
    return np.sqrt(np.mean((img1 - img2) ** 2))


def test(model, args):
    """Test the model on a dataset
    
    Args:
        model: The model to test
        args: Command line arguments
    
    Returns:
        Dictionary with test results
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    # Create dataset and dataloader
    dataset = TestDataset(lq_dir=args.lq_path, gt_dir=args.gt_path, task=args.task, scale=args.scale)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)
    
    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Metrics
    psnr_values = []
    ssim_values = []
    rmse_values = []
    
    with torch.no_grad():
        for lq, gt, filename in tqdm(dataloader, desc="Testing"):
            # Move to device
            lq = lq.to(device)
            gt = gt.to(device)
            
            # Make image dimensions divisible by patch_size
            original_size = lq.shape[2:]
            lq = _make_tensor_divisible(lq, args.patch_size)
            
            # Forward pass
            output = model(lq)
            
            # Crop to original size
            output = output[:, :, :original_size[0]*args.scale, :original_size[1]*args.scale]
            
            # Convert to numpy for metrics
            output_np = output.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            gt_np = gt.squeeze(0).cpu().numpy().transpose(1, 2, 0)
            
            # Calculate metrics
            psnr = calculate_psnr(output_np, gt_np)
            ssim = calculate_ssim(output_np, gt_np)
            rmse = calculate_rmse(output_np, gt_np)
            
            psnr_values.append(psnr)
            ssim_values.append(ssim)
            rmse_values.append(rmse)
            
            # Save result images
            if args.save_images:
                # Get filename
                name = filename[0]
                base_name, ext = os.path.splitext(name)
                
                # Convert to uint8
                output_uint8 = (output_np * 255.0).clip(0, 255).astype(np.uint8)
                lq_uint8 = (lq.squeeze(0).cpu().numpy().transpose(1, 2, 0) * 255.0).clip(0, 255).astype(np.uint8)
                gt_uint8 = (gt_np * 255.0).clip(0, 255).astype(np.uint8)
                
                # Save images
                cv2.imwrite(os.path.join(args.save_path, f"{base_name}_result{ext}"), 
                           cv2.cvtColor(output_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.save_path, f"{base_name}_input{ext}"), 
                           cv2.cvtColor(lq_uint8, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(args.save_path, f"{base_name}_gt{ext}"), 
                           cv2.cvtColor(gt_uint8, cv2.COLOR_RGB2BGR))
    
    # Calculate average metrics
    avg_psnr = sum(psnr_values) / len(psnr_values)
    avg_ssim = sum(ssim_values) / len(ssim_values)
    avg_rmse = sum(rmse_values) / len(rmse_values)
    
    # Save results to file
    results_path = os.path.join(args.save_path, "results.txt")
    with open(results_path, 'w') as f:
        f.write(f"MambaIRv2 Test Results (X{args.scale})\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.lq_path}\n")
        f.write(f"Number of images: {len(psnr_values)}\n")
        f.write(f"Average PSNR: {avg_psnr:.4f}\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n")
        f.write(f"Average RMSE: {avg_rmse:.4f}\n")
    
    print(f"Results saved to: {results_path}")
    print(f"Average PSNR: {avg_psnr:.4f}")
    print(f"Average SSIM: {avg_ssim:.4f}")
    print(f"Average RMSE: {avg_rmse:.4f}")
    
    return {
        'avg_psnr': avg_psnr,
        'avg_ssim': avg_ssim,
        'avg_rmse': avg_rmse,
        'num_images': len(psnr_values)
    }


def discover_test_datasets(root_dir, scale=4):
    """Discover test datasets in the directory structure
    
    Args:
        root_dir: Root directory containing test data
        scale: Super-resolution scale factor (fixed to 4)
    
    Returns:
        List of dataset dictionaries with name, lq_path, gt_path
    """
    scale = int(scale)  # Force scale to be integer
    if scale != 4:
        print(f"Warning: Scale factor forced to 4 (requested {scale})")
    
    datasets = []
    
    # Check if root_dir exists
    if not os.path.isdir(root_dir):
        print(f"Root directory does not exist: {root_dir}")
        return datasets
    
    # Look for dataset directories (both direct and nested)
    for item in os.listdir(root_dir):
        item_path = os.path.join(root_dir, item)
        if not os.path.isdir(item_path):
            continue
            
        # Check for direct LQ/GT structure
        lq_path = os.path.join(item_path, 'LQ')
        gt_path = os.path.join(item_path, 'GT')
        
        if os.path.isdir(lq_path) and os.path.isdir(gt_path):
            datasets.append({
                'name': item,
                'lq_path': lq_path,
                'gt_path': gt_path
            })
            print(f"Found dataset: {item}")
            continue
        
        # Check for nested structure with scale
        scale_path = os.path.join(item_path, f'X{scale}')
        if os.path.isdir(scale_path):
            lq_path = os.path.join(scale_path, 'LQ')
            gt_path = os.path.join(scale_path, 'GT')
            
            if os.path.isdir(lq_path) and os.path.isdir(gt_path):
                datasets.append({
                    'name': item,
                    'lq_path': lq_path,
                    'gt_path': gt_path
                })
                print(f"Found dataset: {item} (X{scale})")
    
    print(f"Discovered {len(datasets)} datasets for X{scale}")
    return datasets


def auto_test_all_datasets(model, args, window_size):
    """Automatically test all discovered datasets
    
    Args:
        model: The model to test
        args: Command line arguments
        window_size: Window size for processing
    """
    # Discover datasets
    datasets = discover_test_datasets(args.test_root, args.scale)
    
    if not datasets:
        print("No test datasets found!")
        return
    
    # Create results directory
    os.makedirs(args.results_root, exist_ok=True)
    
    # Test each dataset
    all_results = []
    for dataset in datasets:
        print(f"\n{'='*50}")
        print(f"Testing dataset: {dataset['name']}")
        print(f"{'='*50}")
        
        # Update paths for this dataset
        args.lq_path = dataset['lq_path']
        args.gt_path = dataset['gt_path']
        args.save_path = os.path.join(args.results_root, f"{dataset['name']}_X{args.scale}")
        
        # Test the dataset
        result = test(model, args)
        
        # Store results
        all_results.append({
            'dataset': dataset['name'],
            'avg_psnr': result['avg_psnr'],
            'avg_ssim': result['avg_ssim'],
            'avg_rmse': result['avg_rmse'],
            'num_images': result['num_images']
        })
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUMMARY OF ALL DATASETS")
    print(f"{'='*50}")
    print(f"{'Dataset':<20} {'PSNR':<8} {'SSIM':<8} {'RMSE':<8} {'Images':<8}")
    print("-" * 60)
    
    total_psnr = 0
    total_ssim = 0
    total_rmse = 0
    total_images = 0
    
    for result in all_results:
        print(f"{result['dataset']:<20} {result['avg_psnr']:<8.4f} {result['avg_ssim']:<8.4f} "
              f"{result['avg_rmse']:<8.4f} {result['num_images']:<8}")
        total_psnr += result['avg_psnr'] * result['num_images']
        total_ssim += result['avg_ssim'] * result['num_images']
        total_rmse += result['avg_rmse'] * result['num_images']
        total_images += result['num_images']
    
    if total_images > 0:
        print("-" * 60)
        print(f"{'OVERALL':<20} {total_psnr/total_images:<8.4f} {total_ssim/total_images:<8.4f} "
              f"{total_rmse/total_images:<8.4f} {total_images:<8}")
    
    # Save summary to file
    summary_path = os.path.join(args.results_root, f"summary_X{args.scale}.txt")
    with open(summary_path, 'w') as f:
        f.write(f"MambaIRv2 Test Results Summary (X{args.scale})\n")
        f.write(f"{'='*50}\n")
        f.write(f"Dataset: {args.model_size}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"{'='*50}\n\n")
        
        f.write(f"{'Dataset':<20} {'PSNR':<8} {'SSIM':<8} {'RMSE':<8} {'Images':<8}\n")
        f.write("-" * 60 + "\n")
        
        for result in all_results:
            f.write(f"{result['dataset']:<20} {result['avg_psnr']:<8.4f} {result['avg_ssim']:<8.4f} "
                   f"{result['avg_rmse']:<8.4f} {result['num_images']:<8}\n")
        
        if total_images > 0:
            f.write("-" * 60 + "\n")
            f.write(f"{'OVERALL':<20} {total_psnr/total_images:<8.4f} {total_ssim/total_images:<8.4f} "
                   f"{total_rmse/total_images:<8.4f} {total_images:<8}\n")
    
    print(f"\nSummary saved to: {summary_path}")


def main():
    """Main function for testing MambaIRv2 model"""
    parser = argparse.ArgumentParser(description='Test MambaIRv2 model for super-resolution')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, default='pretrained/mambairv2_x4.pth',
                        help='Path to the model checkpoint')
    parser.add_argument('--model_size', type=str, default='medium', choices=['tiny', 'light', 'medium'],
                        help='Model size variant')
    
    # Task parameters
    parser.add_argument('--task', type=str, default='classical_sr',
                        choices=['classical_sr', 'real_sr', 'gray_sr'],
                        help='Task type')
    parser.add_argument('--scale', type=int, default=4,
                        help='Super-resolution scale factor (fixed to 4)')
    parser.add_argument('--patch_size', type=int, default=64,
                        help='Patch size for processing')
    
    # Data parameters
    parser.add_argument('--lq_path', type=str, default=None,
                        help='Path to low-quality images directory')
    parser.add_argument('--gt_path', type=str, default=None,
                        help='Path to ground-truth images directory')
    parser.add_argument('--test_root', type=str, default='test_data',
                        help='Root directory for test datasets')
    
    # Output parameters
    parser.add_argument('--save_path', type=str, default='mambairv2_x4',
                        help='Path to save results')
    parser.add_argument('--results_root', type=str, default='mambairv2_x4',
                        help='Root directory for all results')
    parser.add_argument('--save_images', action='store_true',
                        help='Save output images')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Force scale to be 4
    if args.scale != 4:
        print(f"Warning: Scale factor forced to 4 (requested {args.scale})")
        args.scale = 4
    
    # Create output directory
    os.makedirs(args.save_path, exist_ok=True)
    
    # Define model
    print(f"Defining MambaIRv2 model (size: {args.model_size})")
    model, window_size = define_model(args)
    
    # Load checkpoint
    if os.path.exists(args.model_path):
        print(f"Loading checkpoint from {args.model_path}")
        checkpoint = torch.load(args.model_path, map_location='cpu')
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")
    else:
        print(f"Warning: Checkpoint not found at {args.model_path}")
        print("Using randomly initialized model")
    
    # Test mode
    if args.lq_path and args.gt_path:
        # Test specific dataset
        print(f"Testing specific dataset: {args.lq_path}")
        test(model, args)
    else:
        # Auto-discover and test all datasets
        print(f"Auto-discovering datasets in {args.test_root}")
        auto_test_all_datasets(model, args, window_size)
    
    print("Testing completed!")


if __name__ == '__main__':
    main()