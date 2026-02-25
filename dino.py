# Advanced pretrained models: DINO (Vision Transformer with self-supervised learning)

import timm
import torch.nn as nn

class DINOKeypointDetector(nn.Module):
    def __init__(self, num_keypoints=68, model_name='vit_base_patch16_224.dino', pretrained=True, freeze_backbone=True):
        """
        DINO Vision Transformer-based keypoint detector with transfer learning.
        
        Args:
            num_keypoints: Number of keypoints (68 for facial keypoints)
            model_name: DINO model variant (e.g., 'vit_base_patch16_dinov3.lvd1689m', 'vit_small_patch16_dinov3.lvd1689m')
            pretrained: Whether to use pretrained DINO weights
            freeze_backbone: Whether to freeze backbone during initial training
        """
        super().__init__()
        
        # Load pretrained DINO model from timm
        self.backbone = timm.create_model(model_name, pretrained=pretrained)
        
        # Get the output feature dimension from DINO
        # DINO outputs cls token features (typically 768 for base, 384 for small)
        self.backbone_out_features = self.backbone.embed_dim
        
        # Remove the classification head
        self.backbone.head = nn.Identity()
        
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
        # Handle grayscale input: convert to RGB by repeating channels
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)  # [batch, 1, H, W] → [batch, 3, H, W]
        
        # DINO forward pass - returns cls token features
        features = self.backbone.forward_features(x)  # [batch_size, embed_dim]
        
        # If features have extra dimensions (e.g., [batch, seq_len, embed_dim]), take cls token
        if len(features.shape) > 2:
            features = features[:, 0, :]  # Take cls token
        
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
