from pathlib import Path
import sys
import copy
import os
import yaml
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from mmcv import Config
import matplotlib.pyplot as plt
from PIL import Image
from torchsummary import summary
import cv2

from . import utils
from .model.HamNet import HamNet
from data.pascal_voc12 import PascalVOC
from .sync_bn.nn.modules import SynchronizedBatchNorm2d

@torch.no_grad()
def validation(model, val_loader, n_cls, device = 'cpu'):
    mious = []
    for batch in val_loader:
        image, label= batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1).to(device)
        img_cls_map = utils.inference(model, n_cls, image, 512, 512)
        miou = utils.miou(img_cls_map.to(device), label, n_cls)
        mious.append(miou)

    return np.mean(mious)

@torch.no_grad()
def plot_color_map(model, loader, n_cls, palette, output_dir, device = 'cpu'):
    for i, batch in enumerate(loader):
        image, label= batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1)
        img_cls_map = utils.inference(model, n_cls, image,512, 512)
        label_color_map = np.zeros(image.shape[1:]).transpose(1, 2, 0)
        pred_color_map = np.zeros(image.shape[1:]).transpose(1, 2, 0)
        for cls in range(n_cls):
            color = palette[cls]
            indices_label = label == cls
            label_color_map[indices_label[0], :] = color
            indices_pred = img_cls_map == cls
            pred_color_map[indices_pred[0], :] = color
        fig, axs = plt.subplots(1, 3)
        axs[0].imshow(image[0].permute(1, 2, 0).cpu())
        axs[0].set_title('input image')
        axs[1].imshow(label_color_map)
        axs[1].set_title('ground truth')
        axs[2].imshow(pred_color_map)
        axs[2].set_title('predicted')
        axs[0].axis('off')
        axs[1].axis('off')
        axs[2].axis('off')

        fig_path = output_dir / f'{i + 1}.jpg'
        plt.savefig(fig_path)
        plt.close()



def train(num_layers, lr, num_epochs = 10,  batch_size = 8, device = 'cpu', parallel = False):
    # Create the dataset
    train_dataset = PascalVOC('train')
    val_dataset = PascalVOC('val')
    n_cls = len(train_dataset.classes)
    palette = copy.deepcopy(train_dataset.palette)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers=4)
    hamNet = HamNet(n_cls, num_layers)
    weights = torch.ones(n_cls, device = device)
    weights[0] = 0.3
    criterion = nn.CrossEntropyLoss(weight = weights, ignore_index = 255)
    #optimizer = optim.SGD(hamNet.parameters(), lr = lr, momentum = 0.9, weight_decay = 0.0005)
    optimizer = optim.Adam(hamNet.parameters(), lr = lr, weight_decay = 0.0005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size = 1, gamma = 0.1)
    hamNet.to(device)
    if parallel and torch.cuda.device_count() > 1:
        hamNet = nn.DataParallel(hamNet)

        

    
    #print(summary(hamNet, (3, 512, 512)))
    # print(hamNet.module)
    
    best_miou = 0
    best_model = None
    #torch.autograd.set_detect_anomaly(True)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        hamNet.train()
        running_loss = 0.0
        # if (epoch + 1) == 5:
        #     if parallel and torch.cuda.device_count() > 1:
        #         for parm in hamNet.module.backbone.parameters():
        #             parm.requires_grad = True
        #     else:
        #         for parm in hamNet.backbone.parameters():
        #             parm.requires_grad = True

        for i, batch in enumerate(train_loader):
            images, labels = batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1).to(device)
            optimizer.zero_grad()
            outputs = hamNet(images)
            loss = criterion(outputs + 1e-6, labels).double()
            #loss = loss.clamp(min=-1.0, max=1.0)
            loss.backward()

            # for name, param in hamNet.module.named_parameters():
            #     if param.grad is not None and torch.isnan(param.grad).any():
            #         print(name)
            #         print(param.grad)
            #         sys.exit()
            #print(torch.isnan(list(hamNet.parameters())[0].grad).any())
            
            torch.nn.utils.clip_grad_norm_(hamNet.parameters(), max_norm=1.0)
            
            optimizer.step()
            running_loss += loss.item() * images.size(0)
            if i % 10 == 9:
                print('[Epoch {}, Batch {}] Loss: {:.4f}'.format(epoch+1, i+1, running_loss/((i+1)*8)))

        scheduler.step()

        if (epoch + 1) % 5 == 0:
            hamNet.eval()
            miou = validation(hamNet, val_loader, n_cls, device = device)
            print(f'The validation miou is {miou}' )

            if miou > best_miou:
                best_miou = miou
                best_model_dict = hamNet.state_dict()
        

    print(f'The best validation miou is {best_miou}' )
    
   
    model_dir = Path(__file__).parent / 'checkpoints'
    os.makedirs(model_dir, exist_ok = True)
    model_path = model_dir / f'HamNet_{num_layers}.pth'
    torch.save(best_model_dict, model_path)

    output_dir = Path(__file__).parent / 'val_outputs' / f'HamNet_{num_layers}'
    os.makedirs(output_dir, exist_ok= True)
    hamNet.load_state_dict(best_model_dict)
    plot_color_map(hamNet, val_loader, n_cls, palette, output_dir, device=device) 
    return best_miou

if __name__ == '__main__':
    utils.reproduce()
    # parse the config
    parser = argparse.ArgumentParser(description='Train the HamNet')
    # parse lr
    parser.add_argument('--lr', type = float, default = 1e-6, help = 'The learning rate')
    # parse num_epochs
    parser.add_argument('--num_epochs', type = int, default = 10, help = 'The number of epochs')
    # parse batch_size
    parser.add_argument('--batch_size', type = int, default = 8, help = 'The batch size')
    # parse device
    parser.add_argument('--device', type = str, default = 'cpu', help = 'The device to use')
    parser.add_argument('--parallel', type = bool, default = False, help = 'Whether to use parallel')
    parser.add_argument('--num_layers', type = int, default = 50, help = 'The number of resnet layers')
    args = parser.parse_args()

    if torch.cuda.is_available() and 'cuda' in args.device:
        device = args.device
    else:
        device = 'cpu'
        print('Use cpu')

    best_miou = train(args.num_layers, lr = args.lr, num_epochs = args.num_epochs, batch_size = args.batch_size, device = device, parallel = args.parallel)
    result_path = Path(__file__).parent / 'checkpoints' / f'ResNet{args.num_layers}.txt'
    with open(result_path, 'w') as f:
        f.write(f'The best miou is {best_miou}\n')
        f.write(f'The learning rate is {args.lr}\n')
        f.write(f'The number of epochs is {args.num_epochs}\n')
        f.write(f'The batch size is {args.batch_size}\n')
    
    
        

