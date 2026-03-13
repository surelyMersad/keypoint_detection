import os
import pandas as pd
import torch
from torch.utils.data import Dataset
import cv2
from scipy.ndimage import gaussian_filter
import numpy as np
import matplotlib.image as mpimg

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.index[idx])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, :].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}

        if self.transform:
            sample = self.transform(sample)

        return sample




class FacialKeypointsHeatmapDataset(Dataset):
    """Face Landmarks dataset with heatmap generation."""

    def __init__(self, csv_file, root_dir, transform=None, output_size=64, sigma=1, image_size=224):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
            output_size (int): Size of the output heatmaps (default: 64x64)
            sigma (float): Standard deviation for Gaussian kernel (default: 1)
            image_size (int): Size of the input image after transforms (default: 224)
        """
        self.key_pts_frame = pd.read_csv(csv_file, index_col=0)
        self.root_dir = root_dir
        self.transform = transform
        self.output_size = output_size
        self.sigma = sigma
        self.image_size = image_size

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.index[idx])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        key_pts = self.key_pts_frame.iloc[idx, :].values
        key_pts = key_pts.astype("float").reshape(-1, 2)
        sample = {"image": image, "keypoints": key_pts}

        if self.transform:
            sample = self.transform(sample)

        # Generate heatmaps
        heatmaps = self.generate_heatmaps(sample["keypoints"])
        sample["heatmaps"] = heatmaps
        
        return sample

    
    def generate_heatmaps(self, keypoints):
        """
        Generate heatmaps for each keypoint
        Args:
            keypoints: Tensor or numpy array of shape (68, 2) for 68 keypoints with (x, y) coordinates
        Returns:
            heatmaps: Tensor of shape (68, output_size, output_size)
        """
        # Convert keypoints to numpy if it's a tensor
        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.numpy()
        
        num_keypoints = keypoints.shape[0]
        heatmaps = np.zeros((num_keypoints, self.output_size, self.output_size), dtype=np.float32)
        
       
        keypoints_scaled = (keypoints * 50 + 100) * (self.output_size / self.image_size)
        
        # Generate a heatmap for each keypoint
        for i in range(num_keypoints):
            # Get the scaled coordinates
            x, y = keypoints_scaled[i]
            
            # Skip if keypoint is invalid
            if np.isnan(x) or np.isnan(y):
                continue
            
            # Convert to int for indexing
            x_int, y_int = max(0, min(self.output_size-1, int(x))), max(0, min(self.output_size-1, int(y)))
            
            # Create a single hot pixel
            heatmap = np.zeros((self.output_size, self.output_size), dtype=np.float32)
            heatmap[y_int, x_int] = 1.0
            
            # Apply gaussian filter to create a soft heatmap
            heatmap = gaussian_filter(heatmap, sigma=self.sigma)
            heatmap = heatmap/heatmap.max()
            
            # Normalize heatmap to 0-1 range
            if heatmap.max() > 0:
                heatmap = heatmap / heatmap.max()
                
            heatmaps[i] = heatmap
        
        return torch.from_numpy(heatmaps)
