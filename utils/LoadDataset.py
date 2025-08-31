"""
Dataset Loading Utility for Nano-U

This module provides a PyTorch Dataset implementation for loading and preprocessing
image and mask pairs for training and evaluation of neural network models.
"""

import cv2
import torch
from torch.utils.data import Dataset

class LoadDataset(Dataset):
    """
    PyTorch Dataset for loading image and mask pairs.
    
    This dataset loads image-mask pairs, with options for applying transformations
    during loading. It handles sorting based on frame numbers in the filenames.
    """
    
    def __init__(self, img_files, mask_files, transform=None):
        """
        Initialize the dataset with image and mask file paths.
        
        Args:
            img_files (list): List of paths to image files
            mask_files (list): List of paths to corresponding mask files
            transform (callable, optional): Optional transform to be applied to samples
        """
        def get_frame_number(filename):
            # Extract the frame number from filename like 'd5_s1_frame123.png' or 'mask123.png'
            import re
            if 'frame' in filename:
                match = re.search(r'frame(\d+)', filename)
            else:
                match = re.search(r'mask(\d+)', filename)
            return int(match.group(1)) if match else 0
            
        # Sort both lists numerically by frame/mask number
        self.img_files = sorted(img_files, key=get_frame_number)
        self.mask_files = sorted(mask_files, key=get_frame_number)
        self.transform = transform

    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.img_files)

    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        
        Args:
            idx (int): Index of the sample to retrieve
            
        Returns:
            tuple: (image, mask) pair as PyTorch tensors
            
        Raises:
            RuntimeError: If there's an issue loading the image or mask
        """
        try:
            img = cv2.imread(self.img_files[idx])[:, :, ::-1]  # Convert BGR to RGB
            mask = cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE)  # Load as grayscale
        except Exception as e:
            raise RuntimeError(f"Error loading image or mask at index {idx}: {e}")
            
        img = torch.tensor(img.copy(), dtype=torch.float32).permute(2, 0, 1) / 255.0  # Normalize to [0, 1]
        mask = torch.tensor(mask, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize to [0, 1]
              
        return img, mask