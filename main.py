import time
import torch
import torch.nn as nn
import torch.optim as optim
from utils.utils import *
from model.model import SegFormer
import multiprocessing
import albumentations as A
from albumentations.pytorch import ToTensorV2
from utils.loss import FocalLoss
import train
from torchvision.datasets import Cityscapes
import os

torch.manual_seed(0)
LEARNING_RATE = 0.001
NUM_WORKERS = multiprocessing.cpu_count()//4
EPOCHS = 100

EXP_NAME = 'segformer'

SAVE_PATH = os.path.join("log_dir",EXP_NAME)
def main():
    train_transform = A.Compose([
    A.Resize(256, 512),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.1),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),]) 
    val_transform = A.Compose([
    A.Resize(256, 512),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),])
    
    print('data loaded successfully')
    train_loader = get_cityscapes_data(root_dir = 'data',mode='fine', split='train', num_workers = NUM_WORKERS, batch_size = 4, transforms = train_transform, shuffle=True)
    val_loader = get_cityscapes_data(root_dir = 'data', mode='fine', split='val', num_workers = NUM_WORKERS, batch_size = 1, transforms = val_transform, shuffle=True)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("Device", device, "found sucessfully!")
    # model = UNET(in_channels=3, out_channels=19)
    model = SegFormer(
        in_channels=3,
        widths=[64, 128, 256, 512],
        depths=[3, 4, 6, 3],
        all_num_heads=[1, 2, 4, 8],
        patch_sizes=[7, 3, 3, 3],
        overlap_sizes=[4, 2, 2, 2],
        reduction_ratios=[8, 4, 2, 1],
        mlp_expansions=[4, 4, 4, 4],
        decoder_channels=256,
        scale_factors=[8, 4, 2, 1],
        num_classes=19,
    )
    print("Model loaded")
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',factor=0.2, patience=2)
    alpha = torch.tensor([0.8373, 0.9180, 0.8660, 1.0345, 1.0166, 0.9969, 0.9754, 1.0489, 0.8786, 1.0023, 0.9539, 0.9843, 1.1116, 0.9037, 1.0865, 1.0955, 1.0865, 1.1529, 1.0507]).to(device)
    criterion = FocalLoss(alpha = alpha, gamma = 2, ignore_index = 255)
    print("Optimizer and Loss defined")
    train.train_model(num_epochs=EPOCHS, model=model, device=device, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, loss_function=criterion, scheduler=scheduler, save_path = SAVE_PATH)
if __name__ == '__main__':
    main()