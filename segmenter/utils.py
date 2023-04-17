import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import yaml


def reproduce(seed = 1):
    global device
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_configs():
    configs = yaml.load('./configs.yaml')
    return configs

def miou(predicted, target, num_classes):
    ious = []
    eps = 1e-6

    for cls in range(num_classes):
        pred_mask = predicted == cls
        target_mask = target == cls
        intersection = (pred_mask & target_mask).float().sum()
        union = (pred_mask | target_mask).float().sum()
        iou = (intersection + eps) / (union + eps)
        ious.append(iou)
    ious = torch.tensor(ious)
    return torch.mean(ious).numpy()

def resize(image, size):
    h, w = image.shape[2], image.shape[3]
    if h < w:
        ratio = w / h
        h_res, w_res = size, ratio * size
    else:
        ratio = h / w
        h_res, w_res = ratio * size, size
    
    if min(h, w) < size:
        im_res = F.interpolate(image, (int(h_res), int(w_res)), mode = 'bilinear')
    
    else:
        im_res = image
    
    return im_res

def sliding_window(image, window_size, window_stride):
    H, W = image.shape[2], image.shape[3]

    windows = {'crop': [], 'anchors': []}
    h_anchors = torch.arange(0, H, window_stride)
    w_anchors = torch.arange(0, W, window_stride)
    h_ws = [h.item() for h in h_anchors if h < H - window_size] + [H - window_size]
    w_ws = [w.item() for w in w_anchors if w < W - window_size] + [W - window_size]
    for h in h_ws:
        for w in w_ws:
            window = image[:, :, h:h + window_size, w:w + window_size]
            windows['crop'].append(window)
            windows['anchors'].append((h, w))
    windows['shape'] = (H, W)
    return windows

def merge_windows(windows, window_size):
    seg_maps = windows['seg_maps']
    anchors = windows['anchors']
    n_cls = seg_maps.shape[1]
    H, W = windows['shape']

    logits = torch.zeros((n_cls, H, W))
    count = torch.zeros((1, H, W))
    for map, (h, w) in zip(seg_maps, anchors):
        logits[:, h:h + window_size, w:w + window_size] += map
        count[:, h:h + window_size, w:w + window_size] += 1
    logits = logits / count
    # logits = F.interpolate(logits.unsqueeze(0), ori_shape, mode = 'bilinear')[0]
    return logits

def inference(model, n_cls, image, window_size, window_stride):
    image = resize(image, window_size)
    windows = sliding_window(image, window_size, window_stride)
    crops = torch.stack(windows['crop'])[:, 0]
    num_crops = len(crops)
    seg_maps = torch.zeros((num_crops, n_cls, window_size, window_size))
    with torch.no_grad():
        for i in range(0, num_crops):
            seg_maps[i: i + 1] = model(crops[i : i + 1])
    
    windows['seg_maps'] = seg_maps
    img_seg_map = merge_windows(windows, window_size)
    img_cls_map = img_seg_map.argmax(dim = 0, keepdim = True)
    return img_cls_map







        