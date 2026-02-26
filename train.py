# TODO: Pretrained ResNet backbone
from torchvision import models
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
import wandb
import argparse


from utils import evaluate, evaluate_heatmap, get_training_args, load_model, save_checkpoint, load_checkpoint, visualize_keypoints, load_dataset


# Main Training Loop
step = 0
running_loss = 0

def train(model, train_loader, test_loader, optimizer, criterian, device, model_name='model', num_epochs=10, eval_interval=10, log_interval=5, save_interval=30):
    step = 0
    running_loss = 0

    for epoch in range(num_epochs):
        model.train()

        for batch in train_loader :
            step += 1
            image, keypoints = batch['image'], batch['keypoints']

            # move to device
            image = image.to(device)
            keypoints = keypoints.to(device)

            # reshape keypoints from (batch, 68, 2) -> (batch, 136)
            keypoints = keypoints.reshape(image.size(0), 68 * 2).float()

            # forward pass through the model
            preds = model(image)

            # compute loss
            loss = model.compute_loss(preds, keypoints, criterian)

            # backward pass + optimizer step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                wandb.log({"train_loss": avg_loss, "step": step})
                print(f"Step {step}: Train Loss = {avg_loss:.4f}")
                running_loss = 0

            # Evaluate every eval_interval steps
            if step % eval_interval == 0:
                model.eval()
                val_loss = evaluate(model, test_loader, criterian, device)
                wandb.log({"val_loss": val_loss, "step": step})

                # switch back to train
                model.train()

            # Save checkpoint every save_interval steps
            if step % save_interval == 0:
                save_checkpoint(model, optimizer, epoch, step, model_name,
                                path=f"checkpoints-{model_name}/step_{step}.pth")

    print("✓ Stage 1 completed: Frozen backbone training done!")
    return epoch, step


def train_heatmap(model, train_loader, test_loader, optimizer, device, model_name='unet', num_epochs=10, eval_interval=10, log_interval=5, save_interval=30):
    step = 0
    running_loss = 0
    best_val_loss = float('inf')
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()

        for batch in train_loader:
            step += 1
            images = batch['image'].to(device)
            heatmaps_gt = batch['heatmaps'].to(device)

            heatmaps_pred = torch.sigmoid(model(images)) 
            loss = criterion(heatmaps_pred, heatmaps_gt)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_loss += loss.item()

            if step % log_interval == 0:
                avg_loss = running_loss / log_interval
                wandb.log({"train_loss": avg_loss, "step": step, "epoch": epoch, "lr": optimizer.param_groups[0]['lr']})
                print(f"Epoch {epoch}, Step {step}: Train Loss = {avg_loss:.6f} (lr={optimizer.param_groups[0]['lr']:.6f})")
                running_loss = 0

            if step % eval_interval == 0:
                val_loss = evaluate_heatmap(model, test_loader, device)
                wandb.log({"val_loss": val_loss, "step": step, "epoch": epoch})
                print(f"Epoch {epoch}, Step {step}: Val Loss = {val_loss:.6f}")
                model.train()

                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    save_checkpoint(model, optimizer, epoch, step, model_name,
                                    path=f"checkpoints-{model_name}/best.pth")
                    print(f"  -> New best val loss: {val_loss:.6f}")

            # Save checkpoint every save_interval steps
            if step % save_interval == 0:
                save_checkpoint(model, optimizer, epoch, step, model_name,
                                path=f"checkpoints-{model_name}/step_{step}.pth")

        # Step scheduler at end of epoch
        epoch_val_loss = evaluate_heatmap(model, test_loader, device)
        scheduler.step(epoch_val_loss)
        print(f"--- Epoch {epoch} done. Val Loss = {epoch_val_loss:.6f}, LR = {optimizer.param_groups[0]['lr']:.6f} ---")
        model.train()

    print("Heatmap training complete!")
    return epoch, step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, help='Model name : cnn, resnet, dino')
    parser.add_argument("--criterion", type=str, help="Loss function (mse, Smoothl1Loss)")
    parser.add_argument("--freeze", action="store_true", help="Freeze backbone and only train regression head")
    parser.add_argument("--visualize", action="store_true", help="Load checkpoint and run visualization")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to checkpoint file")

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.checkpoint is None and args.model:
        args.checkpoint = f"checkpoints-{args.model}/last_checkpoint.pth"

    if args.visualize:
        # Load checkpoint and visualize
        model, ckpt = load_checkpoint(args.checkpoint, device=device)
        _, test_dataset = load_dataset()
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
        visualize_keypoints(test_loader, model, device=device)
    else:
        model = load_model(args.model).to(device)
        train_loader, test_loader, optimizer = get_training_args(args.model, model, args.freeze)

        wandb.init(project="facial-keypoints", name=f"{args.model}-train", reinit=True)
        wandb.config.update({
            "model": args.model,
            "criterion": args.criterion if args.model != 'unet' else "mse",
            "freeze": args.freeze,
            "num_params": sum(p.numel() for p in model.parameters()),
        })

        if args.model == 'unet':
            epoch, step = train_heatmap(model, train_loader, test_loader, optimizer, device, model_name=args.model, num_epochs=args.epochs)
        else:
            epoch, step = train(model, train_loader, test_loader, optimizer, args.criterion, device, model_name=args.model, num_epochs=args.epochs)

        # Save last checkpoint
        save_checkpoint(model, optimizer, epoch, step, args.model, path=args.checkpoint)
        wandb.finish()
