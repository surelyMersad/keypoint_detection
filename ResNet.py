from torchvision import models
import torch.nn as nn
import torch

class ResNetKeypointDetector(nn.Module):
    def __init__(self, num_keypoints=68, backbone='resnet18', pretrained=True, freeze_backbone=True):
        """
        ResNet-based keypoint detector with transfer learning.
        
        Args:
            num_keypoints: Number of keypoints (68 for facial keypoints)
            backbone: 'resnet18' or 'resnet34'
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze backbone during initial training
        """
        super().__init__()
        
        # Load pretrained ResNet
        if backbone == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
        elif backbone == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)

        
        # Modify first conv layer to handle grayscale input (1 channel -> 3 channels)
        # Keep pretrained weights by expanding the channel dimension
        original_conv = self.backbone.conv1
        self.backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Initialize by repeating the first channel weights across all 3 channels
        if pretrained:
            with torch.no_grad():
                self.backbone.conv1.weight[:, 0, :, :] = original_conv.weight[:, 0, :, :]
        
        # Get the output features from ResNet's avgpool (512 for resnet18/34)
        self.backbone_out_features = self.backbone.fc.in_features
        
        # Remove the classification head
        self.backbone.fc = nn.Identity()
        
        # Add regression head for keypoint prediction
        self.regression_head = nn.Sequential(
            nn.Linear(self.backbone_out_features, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_keypoints * 2)  # Output: 68 keypoints * 2 coordinates
        )
        
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            self._freeze_backbone()
    
    def _freeze_backbone(self):
        """Freeze all backbone layers for initial training."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def _unfreeze_backbone(self):
        """Unfreeze all backbone layers for fine-tuning."""
        for param in self.backbone.parameters():
            param.requires_grad = True
    
    def forward(self, x):
        # Backbone forward pass
        features = self.backbone(x)  # [batch_size, 512]
        
        # Regression head
        keypoints = self.regression_head(features)  # [batch_size, 136]
        
        return keypoints
      
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

