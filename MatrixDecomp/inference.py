import argparse
from pathlib import Path
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from . import utils
from data.pascal_voc12 import PascalVOC
from .model.HamNet import HamNet

@torch.no_grad()
def plot_color_map(hamNet, loader, n_cls, palette, output_dir, device = 'cpu'):
        for i, batch in enumerate(loader):
            image = batch['img'].to(device)
            img_cls_map = utils.inference(hamNet, n_cls, image, 512, 512)
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
        

def inference(num_layers, device = 'cpu'):
    test_dataset = PascalVOC('test')
    n_cls = len(test_dataset.classes)
    palette = test_dataset.palette
    test_loader = DataLoader(test_dataset, batch_size = 1, shuffle = False, num_workers = 4, pin_memory = True)
    model_dir = Path(__file__).parent / 'checkpoints'
    model_path = model_dir / f'HamNet_{num_layers}.pth'
    # remove parallel in state dict
    state_dict = torch.load(model_path, map_location = device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v
    model = HamNet(n_cls, num_layers).to(device)
    model.load_state_dict(new_state_dict)
    model.eval()
    output_dir = Path(__file__).parent / 'test_outputs' / f'HamNet_{num_layers}'
    os.makedirs(output_dir, exist_ok= True)
    plot_color_map(model, test_loader, n_cls, palette, output_dir, device=device)

if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Inference the HamNet")
    
    parser.add_argument('--num_layers', type = int, default = 50, choices=[50, 101], help = 'The number of resnet layers')
    parser.add_argument('--device', type = str, default = 'cpu', choices = ['cpu', 'cuda'], help = 'The device to run the model')

    args = parser.parse_args()

    if torch.cuda.is_available() and 'cuda' in args.device:
        device = args.device
    else:
        device = 'cpu'
        print('Use cpu')
    
    inference(args.num_layers, device)
