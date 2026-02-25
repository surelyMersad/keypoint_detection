import glob
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, utils
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import lr_scheduler
import torch.optim as optim
from tqdm import tqdm

class CNN(nn.Module):

    def __init__(self, num_keypoints=68, dropout=0.5):
        super().__init__()
        
        # Convolutional Layers (batch norm added after the conv layer) - 4 layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)


        # Fully connected layers
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_keypoints * 2)
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):

        # Conv blocks with pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)
        
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.max_pool2d(x, 2)
        
        # Flatten and FC layers
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return self.fc2(x)

    def compute_loss(self, preds, labels, criterian): 
        """
        Compute regression loss for keypoint detection.
        
        Args:
            preds: Model predictions [batch_size, 136]
            labels: Ground truth [batch_size, 68, 2]
            criterion: Loss function (SmoothL1Loss or MSELoss)
        
        Returns:
            loss: Scalar loss value
        """
        
        if criterian == "mse":
            loss_fn = nn.MSELoss()
        elif criterian == "Smoothl1Loss":
            loss_fn = nn.SmoothL1Loss()
        else:
            raise ValueError(f"Unknown criterion: {criterian}")

        return loss_fn(preds, labels)

