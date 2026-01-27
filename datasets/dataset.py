import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from torchvision import transforms

class SRDataset(Dataset):
    def __init__(self, lq_folder, gt_folder, patch_size=128, scale=2, task='classical_sr', noise_level=15, jpeg_quality=40):
        self.lq_folder = lq_folder
        self.gt_folder = gt_folder
        self.patch_size = patch_size
        self.scale = scale
        self.task = task
        self.noise_level = noise_level
        self.jpeg_quality = jpeg_quality
        
        # 获取所有图像文件
        self.image_files = sorted(os.listdir(gt_folder))
        
        # 数据增强
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(90)
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # 读取图像
        img_name = self.image_files[idx]
        img_gt = cv2.imread(os.path.join(self.gt_folder, img_name))
        img_gt = cv2.cvtColor(img_gt, cv2.COLOR_BGR2RGB)
        
        # 根据任务生成低质量图像
        if self.task in ['classical_sr', 'lightweight_sr']:
            # 超分辨率：下采样生成低质量图像
            h, w = img_gt.shape[:2]
            img_lq = cv2.resize(img_gt, (w//self.scale, h//self.scale), interpolation=cv2.INTER_CUBIC)
            # 随机裁剪patch
            if h > self.patch_size * self.scale and w > self.patch_size * self.scale:
                top = random.randint(0, h - self.patch_size * self.scale)
                left = random.randint(0, w - self.patch_size * self.scale)
                img_gt_patch = img_gt[top:top+self.patch_size*self.scale, left:left+self.patch_size*self.scale]
                lq_top = top // self.scale
                lq_left = left // self.scale
                img_lq_patch = img_lq[lq_top:lq_top+self.patch_size, lq_left:lq_left+self.patch_size]
            else:
                img_gt_patch = img_gt
                img_lq_patch = img_lq
        elif self.task in ['gray_dn', 'color_dn']:
            # 去噪：添加高斯噪声
            img_lq = img_gt + np.random.normal(0, self.noise_level/255., img_gt.shape)
            img_lq = np.clip(img_lq, 0, 1)
            # 随机裁剪
            h, w = img_gt.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                img_gt_patch = img_gt[top:top+self.patch_size, left:left+self.patch_size]
                img_lq_patch = img_lq[top:top+self.patch_size, left:left+self.patch_size]
            else:
                img_gt_patch = img_gt
                img_lq_patch = img_lq
        elif self.task == 'jpeg_car':
            # JPEG压缩：模拟JPEG压缩
            img_lq = cv2.imencode('.jpg', img_gt, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])[1]
            img_lq = cv2.imdecode(img_lq, cv2.IMREAD_COLOR)
            img_lq = cv2.cvtColor(img_lq, cv2.COLOR_BGR2RGB)
            # 随机裁剪
            h, w = img_gt.shape[:2]
            if h > self.patch_size and w > self.patch_size:
                top = random.randint(0, h - self.patch_size)
                left = random.randint(0, w - self.patch_size)
                img_gt_patch = img_gt[top:top+self.patch_size, left:left+self.patch_size]
                img_lq_patch = img_lq[top:top+self.patch_size, left:left+self.patch_size]
            else:
                img_gt_patch = img_gt
                img_lq_patch = img_lq
        
        # 数据增强
        img_gt_patch = self.transform(img_gt_patch)
        img_lq_patch = self.transform(img_lq_patch)
        
        return img_lq_patch, img_gt_patch 