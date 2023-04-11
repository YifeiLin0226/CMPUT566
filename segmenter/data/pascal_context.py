import torch
from torch.utils.data import Dataset
from mmseg.datasets import PascalContextDataset

from .config import pascal_context

class PascalContext(Dataset):

    def __init__(self, split):
        self.dataset = PascalContextDataset(**pascal_context.data[split])

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]
        data = [sample['img'].data, sample['gt_semantic_seg'].data]
        return data