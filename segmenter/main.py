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
from torchvision.datasets import ImageNet
from torchvision import transforms
from timm.models import create_model
from tqdm import tqdm
from mmcv import Config
import matplotlib.pyplot as plt
from PIL import Image
import cv2

from . import utils
from .model.segmenter import Segmenter
from data.pascal_voc12 import PascalVOC

@torch.no_grad()
def validation(segmenter, val_loader, n_cls, device = 'cpu'):
    mious = []
    for batch in val_loader:
        image, label, ori_shape = batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1).to(device), batch['img_metas']['ori_shape'][:2]
        img_cls_map = utils.inference(segmenter, n_cls, image, ori_shape, 512, 512)
        miou = utils.miou(img_cls_map.to(device), label, n_cls)
        mious.append(miou)
    return np.mean(mious)

@torch.no_grad()
def plot_color_map(segmenter, loader, n_cls, palette, output_dir, device = 'cpu'):
    for i, batch in enumerate(loader):
        image, label = batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1)
        img_cls_map = utils.inference(segmenter, n_cls, image, 512, 512)
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





def train(model_name, pretrained_link, num_layers, d_model, num_heads, mask, lr, num_epochs = 10,  batch_size = 8, device = 'cpu', parallel = False):
    # Create the dataset
    train_dataset = PascalVOC('train')
    val_dataset = PascalVOC('val')
    n_cls = len(train_dataset.classes)
    palette = copy.deepcopy(train_dataset.palette)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers=4)
    encoder = create_model(pretrained_link, pretrained = True, img_size = 512, num_classes = 0, global_pool='')
    segmenter = Segmenter(encoder, n_cls, d_model, num_layers, num_heads, 512, mask = mask)
    criterion = nn.CrossEntropyLoss(ignore_index = 255)
    optimizer = optim.SGD(segmenter.parameters(), lr = lr, momentum = 0.9, weight_decay = 0.0005)
    segmenter.train()
    segmenter.to(device)
    if parallel and torch.cuda.device_count() > 1:
        segmenter = nn.DataParallel(segmenter)
    
    best_miou = 0
    best_model = None
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
        segmenter.train()
        running_loss = 0.0

        for i, batch in enumerate(train_loader):
            images, labels = batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1).to(device)
            
            optimizer.zero_grad()

            outputs = segmenter(images)
            loss = criterion(outputs, labels)
            loss.backward()


            optimizer.step()
            running_loss += loss.item() * images.size(0)
            if i % 10 == 9:
                print('[Epoch {}, Batch {}] Loss: {:.4f}'.format(epoch+1, i+1, running_loss/((i+1)*8)))
            
        
        segmenter.eval()
        miou = validation(segmenter, val_loader, n_cls, device = device)

        if miou > best_miou:
            best_miou = miou
            best_model = copy.deepcopy(segmenter)

    print(f'The best validation miou is {best_miou}' )
    model_dir = Path(__file__).parent / 'checkpoints'
    os.makedirs(model_dir, exist_ok = True)
    model_path = model_dir / model_name
    torch.save(best_model.state_dict(), model_path)

    output_dir = Path(__file__).parent / 'val_outputs' / model_name
    os.makedirs(output_dir, exist_ok= True)
    plot_color_map(best_model, val_loader, n_cls, palette, output_dir, device=device)
    return best_miou 


if __name__ == '__main__':
    name_dict = {'small_mask': 'segmenter_small_16_mask', 'base_mask': 'segmenter_base_16_mask', 'small_linear': 'segmenter_small_16_linear', 'base_linear': 'segmenter_base_16_linear'}
    # parse the config
    parser = argparse.ArgumentParser(description='Train the segmenter')
    # parse model name
    parser.add_argument('--model_name', type = str, default = 'small_linear', choices = ['small_mask', 'base_mask', 'small_linear', 'base_linear'], help = 'The name of the model')
    # parse lr
    parser.add_argument('--lr', type = float, default = 1e-6, help = 'The learning rate')
    # parse num_epochs
    parser.add_argument('--num_epochs', type = int, default = 10, help = 'The number of epochs')
    # parse batch_size
    parser.add_argument('--batch_size', type = int, default = 8, help = 'The batch size')
    # parse device
    parser.add_argument('--device', type = str, default = 'cpu', help = 'The device to use')
    parser.add_argument('--parallel', type = bool, default = False, help = 'Whether to use parallel')
    args = parser.parse_args()

    if torch.cuda.is_available() and 'cuda' in args.device:
        device = args.device
    else:
        device = 'cpu'
        print('Use cpu')

    # load the config using yaml
    config_path = Path(__file__).parent / 'configs.yaml'
    config = yaml.load(open(config_path, 'r'), Loader=yaml.FullLoader)
    model_name = name_dict[args.model_name]
    params = config['model'][model_name]

    best_miou = train(args.model_name, **params, lr = args.lr, num_epochs = args.num_epochs, batch_size = args.batch_size, device = device, parallel = args.parallel)
    result_path = Path(__file__).parent / 'checkpoints' / f'{args.model_name}.txt'
    with open(result_path, 'w') as f:
        f.write(f'The best miou is {best_miou}\n')
        f.write(f'The learning rate is {args.lr}\n')
        f.write(f'The number of epochs is {args.num_epochs}\n')
        f.write(f'The batch size is {args.batch_size}\n')

    
    
        

