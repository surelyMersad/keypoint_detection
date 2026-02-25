from torch.utils.data import Dataset, DataLoader
import os
import torch
import matplotlib.pyplot as plt
from facial_keypoints_dataset import FacialKeypointsDataset
from custom_transforms import data_transform
from cnn import CNN
from ResNet import ResNetKeypointDetector
from dino import DINOKeypointDetector


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
      
def load_dataset():
    # Load and preprocess dataset
    train_dataset = FacialKeypointsDataset('data/training_frames_keypoints.csv', 'data/training', data_transform)
    test_dataset = FacialKeypointsDataset('data/test_frames_keypoints.csv', 'data/test', data_transform)
    return train_dataset, test_dataset

def get_training_args (model, freeze):
    if model == 'cnn' : 
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

    elif model == 'resnet' or 'dino':
        batch_size = 64

        if freeze :
            # stage1 lr is set to be higher
            lr = 5e-4
            model.freeze_backbone()
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

    else :
        raise ValueError('model is not defined - choose between : CNN, Resnet, Dino')
    
    return train_loader, test_loader, optimizer
    


