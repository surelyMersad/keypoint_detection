"""
Regression without Soft-Argmax (RwSa) Loss

Based on: "Heatmap Regression without Soft-Argmax for Facial Landmark Detection"
(Yang & Yeh, 2025) - https://arxiv.org/abs/2508.14929
"""

import torch
import torch.nn as nn


class MySmoothL1Loss(nn.Module):
    """SmoothL1 on the L2 norm of error vectors."""
    def __init__(self, scale=0.01):
        super().__init__()
        self.scale = scale

    def forward(self, err):
        """err: (..., 2) -> (...)"""
        delta_2 = err.pow(2).sum(dim=-1)
        loss = torch.where(
            delta_2 < self.scale * self.scale,
            0.5 / self.scale * delta_2,
            delta_2.clamp(min=1e-6).sqrt() - 0.5 * self.scale,
        )
        return loss


def keypoints_to_heatmap_coords(keypoints, image_size=224, heatmap_size=64):
    """Convert normalized keypoints to [-1, 1] grid coordinates matching heatmap.

    Args:
        keypoints: (B, N, 2) in normalized space: (pts - 100) / 50
    Returns:
        (B, N, 2) in [-1, 1] grid space
    """
    # Denormalize to image pixel space
    pts = keypoints * 50.0 + 100.0
    # Scale to heatmap pixel space
    pts = pts * (heatmap_size / image_size)
    # Normalize to [-1, 1]
    pts = pts / (heatmap_size - 1) * 2.0 - 1.0
    return pts


class RwSaLoss(nn.Module):
    """Regression without Soft-Argmax structured prediction loss.

    Instead of comparing predicted heatmaps to GT heatmaps (like BCE/MSE),
    this loss directly uses GT coordinates:
    - Encourages high heatmap values near the GT location
    - Penalizes high values far from GT via a distance margin
    - Samples "positive" positions near GT from a Gaussian mask
    """
    def __init__(self, eps=1.0, alpha=1.0, sigma=0.05, n_samples=10, dist='smoothl1'):
        """
        Args:
            eps: Temperature parameter for log-sum-exp
            alpha: Margin weighting coefficient
            sigma: Gaussian sigma for sampling mask (in [-1,1] normalized space)
            n_samples: Number of positions to sample per keypoint
            dist: Distance metric for margin ('smoothl1', 'l1', 'l2')
        """
        super().__init__()
        self.eps = eps
        self.alpha = alpha
        self.sigma = sigma
        self.n_samples = n_samples

        if dist == 'smoothl1':
            self.dist_func = MySmoothL1Loss()
        elif dist == 'l1':
            self.dist_func = lambda err: torch.abs(err).sum(dim=-1)
        elif dist == 'l2':
            self.dist_func = lambda err: err.pow(2).sum(dim=-1)
        else:
            raise ValueError(f"Unknown dist: {dist}")

    def forward(self, heatmap, gt_coords):
        """
        Args:
            heatmap: (B, N, H, W) raw predicted heatmap (logits, no sigmoid)
            gt_coords: (B, N, 2) ground truth in [-1, 1] grid space (x, y)
        Returns:
            scalar loss
        """
        b, n, h, w = heatmap.shape

        # Create coordinate grids in [-1, 1]
        yy = torch.linspace(-1, 1, h, device=heatmap.device)
        xx = torch.linspace(-1, 1, w, device=heatmap.device)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')  # (H, W)

        # Error from each pixel to GT: (B, N, H, W, 2)
        gt_x = gt_coords[..., 0].unsqueeze(-1).unsqueeze(-1)  # (B, N, 1, 1)
        gt_y = gt_coords[..., 1].unsqueeze(-1).unsqueeze(-1)
        err_x = gt_x - xx.unsqueeze(0).unsqueeze(0)  # (B, N, H, W)
        err_y = gt_y - yy.unsqueeze(0).unsqueeze(0)
        err = torch.stack([err_x, err_y], dim=-1)  # (B, N, H, W, 2)

        # Distance margin: pixels far from GT get large margin
        margin = self.dist_func(err)  # (B, N, H, W)
        margin = margin.view(b, n, -1)  # (B, N, H*W)

        # Flatten heatmap
        heatmap_1d = heatmap.view(b, n, -1)  # (B, N, H*W)

        # Gaussian sampling mask centered at GT
        dist_sq = err_x.pow(2) + err_y.pow(2)  # (B, N, H, W)
        mask = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        mask = mask.view(b, n, -1) + 1e-12  # (B, N, H*W)

        # Sample positions from mask (near GT = likely positive)
        D = torch.distributions.Categorical(mask.view(b * n, -1))

        losses = []
        for _ in range(self.n_samples):
            g_sample = D.sample().view(b, n, 1).long()

            # Heatmap value at sampled position (should be high)
            numerator = torch.gather(heatmap_1d, -1, g_sample)  # (B, N, 1)

            # Log-sum-exp of (heatmap + alpha * margin) / eps
            denominator = (heatmap_1d + self.alpha * margin) / self.eps
            M = denominator.max(dim=-1, keepdim=True)[0]
            denominator = M + torch.log(
                torch.sum(torch.exp(denominator - M), dim=-1, keepdim=True)
            )

            losses.append(self.eps * denominator - numerator)

        losses = torch.cat(losses, dim=-1)  # (B, N, n_samples)
        return losses.mean()
