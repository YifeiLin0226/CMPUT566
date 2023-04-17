import argparse
from pathlib import Path
import yaml
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from timm.models import create_model
import matplotlib.pyplot as plt

from . import utils
from .model.segmenter import Segmenter
from data.pascal_voc12 import PascalVOC

@torch.no_grad()
def plot_color_map(segmenter, loader, n_cls, palette, output_dir, device = 'cpu'):
        for i, batch in enumerate(loader):
            image = batch['img'].to(device)
            img_cls_map = utils.inference(segmenter, n_cls, image, 512, 512)
            pred_color_map = np.zeros(image.shape[1:]).transpose(1, 2, 0)
            for cls in range(n_cls):
                color = palette[cls]
                indices_pred = img_cls_map == cls
                pred_color_map[indices_pred[0], :] = color
            fig, axs = plt.subplots(1, 2)
            axs[0].imshow(image[0].permute(1, 2, 0).cpu())
            axs[0].set_title('input image')
            axs[1].imshow(pred_color_map)
            axs[1].set_title('predicted')
            axs[0].axis('off')
            axs[1].axis('off')

            fig_path = output_dir / f'{i + 1}.jpg'
            plt.savefig(fig_path)
            plt.close()

def inference(model_name, pretrained_link, num_layers, d_model, num_heads, mask, device = 'cpu'):
    test_dataset = PascalVOC('test')
    n_cls = len(test_dataset.classes)
    palette = test_dataset.palette
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = True)
    encoder = create_model(pretrained_link, pretrained = False, img_size = 512, num_classes = 0, global_pool='')
    segmenter = Segmenter(encoder, n_cls, d_model, num_layers, num_heads, 512, mask = mask).to(device)
    model_dir = Path(__file__).parent / 'checkpoints'
    model_path = model_dir / model_name
    # remove parallel in state dict
    state_dict = torch.load(model_path, map_location = device)
    new_state_dict = {}
    for k, v in state_dict.items():
        name = k[7:]
        new_state_dict[name] = v
    segmenter.load_state_dict(new_state_dict)
    segmenter.eval()
    output_dir = Path(__file__).parent / 'test_outputs' / model_name
    os.makedirs(output_dir, exist_ok= True)
    plot_color_map(segmenter, test_loader, n_cls, palette, output_dir, device=device)

if __name__ == "__main__":
    name_dict = {'small_mask': 'segmenter_small_16_mask', 'base_mask': 'segmenter_base_16_mask', 'small_linear': 'segmenter_small_16_linear', 'base_linear': 'segmenter_base_16_linear'}
    # Parse arguments
    parser = argparse.ArgumentParser(description="Inference the segmenter")
    parser.add_argument('--model_name', type = str, default = 'small_linear', choices = ['small_mask', 'base_mask', 'small_linear', 'base_linear'], help = 'The name of the model')
    parser.add_argument('--device', type = str, default = 'cpu', help = 'The device to use')

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
    inference(args.model_name, **params, device = device)
    
