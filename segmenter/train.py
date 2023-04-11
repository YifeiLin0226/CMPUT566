from pathlib import Path
import sys


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

# Hyperparameters
lr = 1e-6
momentum = 0.9
weight_decay = 0.0005
num_epochs = 10

torch.autograd.set_detect_anomaly(True)
utils.reproduce()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Create the dataset
train_dataset = PascalVOC('train')
val_dataset = PascalVOC('val')
n_cls = len(train_dataset.classes)
palette = train_dataset.palette

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size = 1, num_workers = 4)


encoder = create_model("hf_hub:timm/vit_base_patch16_384.augreg_in21k_ft_in1k", pretrained = True, img_size = 512, num_classes = 0, global_pool='')
segmenter = Segmenter(encoder, n_cls, 768, 12, 12, 512, mask = True)


criterion = nn.CrossEntropyLoss(ignore_index = 255)
optimizer = optim.SGD(segmenter.parameters(), lr = lr, momentum = momentum, weight_decay = weight_decay)

segmenter.train()
segmenter.to(device)
segmenter = nn.DataParallel(segmenter)

for epoch in range(num_epochs):
    print('Epoch {}/{}'.format(epoch+1, num_epochs))
    print('-' * 10)
    segmenter.train()
    running_loss = 0.0
    mious = np.array([])

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
    with torch.no_grad():
        mious = []
        for i, batch in enumerate(val_loader):
            image, label, ori_shape = batch['img'].to(device), batch['gt_semantic_seg'].squeeze(1).to(device), batch['img_metas']['ori_shape'][:2]
            img_cls_map = utils.inference(segmenter, n_cls, image, ori_shape, 512, 512)
            miou = utils.miou(img_cls_map.to(device), label, n_cls)
            mious.append(miou)
            if epoch == num_epochs - 1:
                label_color_map = np.zeros(image.shape[1:]).transpose(1, 2, 0)
                pred_color_map = np.zeros(image.shape[1:]).transpose(1, 2, 0)
                label = label.cpu().numpy()
                for cls in range(n_cls):
                    color = palette[cls]
                    indices_label = label == cls
                    label_color_map[indices_label[0], :] = color
                    indices_pred = img_cls_map == cls
                    pred_color_map[indices_pred[0], :] = color
                label_color_map = label_color_map.astype(np.uint8)
                pred_color_map = pred_color_map.astype(np.uint8)
                label_color_img = Image.fromarray(label_color_map)
                pred_color_img = Image.fromarray(pred_color_map)
                label_color_img.save('label.jpg')
                pred_color_img.save('pred.jpg')
            
                tmp = cv2.cvtColor(np.float32(pred_color_map), cv2.COLOR_BGR2GRAY)
                _,alpha = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
                b, g, r = cv2.split(np.float32(pred_color_map))
                rgba = [b,g,r, alpha]
                dst = cv2.merge(rgba,4)
                plt.imshow(image[0].cpu().numpy().transpose(1, 2, 0))
                plt.imshow(dst, alpha = 0.2)
                plt.savefig('haha.jpg')
                sys.exit()
        
        print(np.mean(mious))


    
           

    
    epoch_loss = running_loss / len(train_dataset)
    epoch_miou = np.mean(mious)
    print('Epoch Loss: {:.4f}, miou: {:.2%}'.format(epoch_loss, epoch_miou))
    




