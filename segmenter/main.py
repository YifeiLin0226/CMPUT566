from pathlib import Path
import sys
import copy
import os

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
from .data.pascal_voc12 import PascalVOC

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
        image, label, ori_shape = batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1).to(device), batch['img_metas']['ori_shape'][:2]
        img_cls_map = utils.inference(segmenter, n_cls, image, ori_shape, 512, 512)
        label_color_map = np.zeros(image.shape[1:]).transpose(1, 2, 0)
        pred_color_map = np.zeros(image.shape[1:]).transpose(1, 2, 0)
        for cls in range(n_cls):
            color = palette[cls]
            indices_label = label == cls
            label_color_map[indices_label[0], :] = color
            indices_pred = img_cls_map == cls
            pred_color_map[indices_pred[0], :] = color
        fig, axs = plt.subplots(1, 2)
        axs[0].imshow(label_color_map)
        axs[0].set_title('ground truth color map')
        axs[1].imshow(pred_color_map)
        axs[1].set_title('predicted color map')
        axs[0].axis('off')
        axs[1].axis('off')

        fig_path = output_dir / f'{i + 1}.jpg'
        plt.savefig(fig_path)





def train(model_name, pretrained_link, num_layer, d_model, num_heads, mask, lr, num_epochs = 10,  batch_size = 8, device = 'cpu', parallel = False):
    # Create the dataset
    train_dataset = PascalVOC('train')
    val_dataset = PascalVOC('val')
    n_cls = len(train_dataset.classes)
    palette = train_dataset.palette
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size = 1, shuffle = False, num_workers=4)
    encoder = create_model(pretrained_link, pretrained = True, img_size = 512, num_classes = 0, global_pool='')
    segmenter = Segmenter(encoder, n_cls, d_model, num_layer, num_heads, 512, mask = mask)
    criterion = nn.CrossEntropyLoss(ignore_index = 255)
    optimizer = optim.SGD(segmenter.parameters(), lr = lr, momentum = 0.9, weight_decay = 0.0005)
    segmenter.train()
    segmenter.to(device)
    if parallel:
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
    model_path = Path(__file__).parent / 'checkpoints' / model_name
    torch.save(best_model.state_dict(), model_path)

    output_dir = Path(__file__).parent / 'val_outputs' / model_name
    os.makedirs(output_dir, exist_ok= True)
    plot_color_map(segmenter, val_loader, n_cls, palette, output_dir, device=device) 

    
    
        

