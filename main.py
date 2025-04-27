import torch
from data.dataset import Dataset
from model.model import Model
from model.model_v1 import ModelV
from training.train_model import TrainModel
import argparse 
from typing import Optional
import sys
import os


def setup_device():
    """Configure and return the appropriate device"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda:0':
        torch.backends.cudnn.benchmark = True  # Enable cuDNN optimization
    print(f"Using device: {device}")
    return device


def main(root, epoch):
    try:
        # 1. Setup
        device = setup_device()

        #-----------------// Data Loading //------------------#
        
        if not os.path.exists(root):
            raise FileNotFoundError(f"Data directory not found at {os.path.abspath(root)}")

        print("\nLoading dataset...")
        dataset = Dataset(root)
        train_loader, test_loader, class_idx = dataset.create_dataloader()
        X, _ = next(iter(train_loader))
        print(f"Found {len(class_idx)} classes: {list(class_idx.keys())}")

        #------------// Model Initialization //--------------#
        print("\nInitializing model...")
        input_shape = X.shape
        # model = Model(input_size=input_shape, class_idx = class_idx).to(device)
        model = ModelV(input_size = input_shape, class_idx = class_idx, conv_chanels= [16, 32, 64],
                        fc_channels = [128, 256, 512], pool = 'max').to(device)
        
        print(model.num_classes, model.output)
        print(f"Model architecture:\n{model}")

        #----------------// Training Setup //-----------------#
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr = 1e-4,
            weight_decay = 0.01
        )

        #--------------------// Scheduler //---------------------------#
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = 50, eta_min = 1e-5)

        # ----------------// Trainer Configuration //-------------------#
        trainer = TrainModel(
            model = model,
            epochs = epoch,
            device = device,
            optimizer = optimizer,
            train_loader = train_loader,
            test_loader = test_loader,
            early_stopping_patience = 3,
            scheduler= scheduler,
            num_classes = model.num_classes  
        )

        #----------------// Training //-------------------#
        print("\nStarting training...")
        trainer.train_model()

        print("\nTraining completed successfully!")

    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        raise


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type = str, default= 'archive')
    parser.add_argument('--epoch', type= int, default= 10)
    args = parser.parse_args()
    main(root = args.root, epoch = args.epoch)
