from torch.utils.data import Dataset, DataLoader
import os
import subprocess
import zipfile
import torch
import matplotlib.pyplot as plt
from cnn import CNN
from ResNet import ResNetKeypointDetector
from dino import DINOKeypointDetector
from unet import UNet
from torchvision import transforms, utils

# the transforms we defined in Notebook 1 are in the helper file `custom_transforms.py`
from custom_transforms import (
    Rescale,
    RandomCrop,
    NormalizeOriginal,
    ToTensor,
    RandomHorizontalFlip,
    ColorJitter,
)

# the dataset we created in Notebook 1
from facial_keypoints_dataset import FacialKeypointsDataset, FacialKeypointsHeatmapDataset

def save_checkpoint(model, optimizer, epoch, step, model_name, path='checkpoints/last_checkpoint.pth'):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'step': step,
        'model_name': model_name,
    }, path)
    print(f"Checkpoint saved to {path}")


def load_checkpoint(path, device='cpu'):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = load_model(checkpoint['model_name'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    print(f"Loaded checkpoint from {path} (model={checkpoint['model_name']}, epoch={checkpoint['epoch']}, step={checkpoint['step']})")
    return model, checkpoint


def load_model(model_name):
    if model_name == 'dino' :
        model = DINOKeypointDetector(pretrained=True, freeze_backbone=True)

    elif model_name == 'cnn':
        model = CNN()
    
    elif model_name == 'resnet':
        model = ResNetKeypointDetector(backbone='resnet18', pretrained=True, freeze_backbone=True)

    elif model_name == 'unet':
        model = UNet(in_channels=1, num_keypoints=68, heatmap_size=64)

    return model


def evaluate(model, test_loader, criterian, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            keypoints = batch['keypoints'].view(batch['image'].size(0), -1).to(device)
            
            outputs = model(images)
            loss = model.compute_loss(outputs, keypoints, criterian)
            total_loss += loss.item()
            
    return total_loss / len(test_loader)


def visualize_keypoints(test_loader, model, device='cuda'):

  model.eval()
  for i, data in enumerate(test_loader):
      image = data['image'][0]
      images = data['image']

      with torch.no_grad():
          images = images.to(device)
          predictions = model(images)

      # UNet outputs heatmaps (B, 68, H, W); other models output (B, 136)
      if predictions.dim() == 4:
          predictions = heatmaps_to_keypoints(predictions)
      else:
          predictions = predictions.reshape(images.size(0), 68, 2)
      predictions = predictions.cpu().numpy()
      
      pred_kpts = predictions[0]
      gt_kpts = data['keypoints'][0].numpy()
      
      # Denormalize
      pred_kpts_denorm = (pred_kpts * 50) + 100
      gt_kpts_denorm = (gt_kpts * 50) + 100
      
      plt.figure(figsize=(10, 6))
      plt.imshow(image.numpy().transpose(1, 2, 0), cmap='gray')
      plt.scatter(pred_kpts_denorm[:, 0], pred_kpts_denorm[:, 1], c='r', s=20, label='Predicted')
      plt.scatter(gt_kpts_denorm[:, 0], gt_kpts_denorm[:, 1], c='g', s=20, label='Ground Truth')
      plt.legend()
      plt.title(f"Test Sample {i}")
      plt.show()
      
      if i >= 4:
          break
      
def download_data(data_dir='data'):
    if os.path.exists(os.path.join(data_dir, 'training')) and os.path.exists(os.path.join(data_dir, 'test')):
        return
    print("Data not found. Downloading...")
    os.makedirs(data_dir, exist_ok=True)
    zip_path = os.path.join(data_dir, 'train-test-data.zip')
    subprocess.run([
        'wget', '-q', '-O', zip_path,
        'https://s3.amazonaws.com/video.udacity-data.com/topher/2018/May/5aea1b91_train-test-data/train-test-data.zip'
    ], check=True)
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(data_dir)
    os.remove(zip_path)
    print("Data downloaded and extracted.")


def load_dataset():
    download_data()
    # defining the data_transform using transforms.Compose([all tx's, . , .])
    # order matters! i.e. rescaling should come before a smaller crop
    data_transform = transforms.Compose(
    [Rescale(250), RandomCrop(224), NormalizeOriginal(), ToTensor()])

    train_dataset = FacialKeypointsDataset('data/training_frames_keypoints.csv', 'data/training', data_transform)
    test_dataset = FacialKeypointsDataset('data/test_frames_keypoints.csv', 'data/test', data_transform)
    return train_dataset, test_dataset


def load_heatmap_dataset(heatmap_size=64):
    download_data()
    train_transform = transforms.Compose([
        Rescale(250), RandomCrop(224), RandomHorizontalFlip(), ColorJitter(),
        NormalizeOriginal(), ToTensor()])
    test_transform = transforms.Compose(
        [Rescale(250), RandomCrop(224), NormalizeOriginal(), ToTensor()])

    train_dataset = FacialKeypointsHeatmapDataset(
        'data/training_frames_keypoints.csv', 'data/training',
        transform=train_transform, output_size=heatmap_size, sigma=4, image_size=224)
    test_dataset = FacialKeypointsHeatmapDataset(
        'data/test_frames_keypoints.csv', 'data/test',
        transform=test_transform, output_size=heatmap_size, sigma=4, image_size=224)
    return train_dataset, test_dataset

def get_training_args (model_name, model, freeze):
    if model_name == 'cnn' :
        batch_size = 64
        lr = 1e-3

        train_dataset, test_dataset = load_dataset()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
        )

        return train_loader, test_loader, optimizer

    elif model_name == 'resnet' or model_name == 'dino':
        batch_size = 64

        if freeze :
            # stage1 lr is set to be higher
            lr = 5e-4
            model._freeze_backbone()
            optimizer = torch.optim.Adam(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
)

        else :
            lr = 1e-4
            model._unfreeze_backbone()

            optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr)


        train_dataset, test_dataset = load_dataset()

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
        )

    elif model_name == 'unet':
        batch_size = 64
        lr = 1e-3

        train_dataset, test_dataset = load_heatmap_dataset()

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    else :
        raise ValueError('model is not defined - choose between : CNN, Resnet, Dino, UNet')

    return train_loader, test_loader, optimizer
    

def heatmaps_to_keypoints(heatmaps, heatmap_size=64, image_size=224):
    """Extract keypoint coordinates from heatmaps using argmax.
    Returns keypoints in normalized space (same as training targets)."""
    batch_size, num_kpts, h, w = heatmaps.shape
    heatmaps_flat = heatmaps.view(batch_size, num_kpts, -1)
    max_indices = heatmaps_flat.argmax(dim=2)

    y_coords = (max_indices // w).float()
    x_coords = (max_indices % w).float()

    # Scale from heatmap space back to image space, then normalize
    x_coords = x_coords * (image_size / heatmap_size)
    y_coords = y_coords * (image_size / heatmap_size)

    # Normalize: (pts - 100) / 50
    x_norm = (x_coords - 100) / 50.0
    y_norm = (y_coords - 100) / 50.0

    keypoints = torch.stack([x_norm, y_norm], dim=2)  # [B, 68, 2]
    return keypoints


def evaluate_heatmap(model, test_loader, device, alpha=10.0):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device)
            heatmaps_gt = batch['heatmaps'].to(device)
            heatmaps_pred = model(images)
            weight = 1.0 + alpha * heatmaps_gt
            loss = (weight * (heatmaps_pred - heatmaps_gt) ** 2).mean()
            total_loss += loss.item()
    return total_loss / len(test_loader)