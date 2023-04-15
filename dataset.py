import torch
from torch.utils.data import Dataset
import numpy as np
import random

class ShapeNet_Dataset(Dataset):
    
    def __init__(self, dataset_path):
        self.dataset = []
        for file_name in sorted(dataset_path.glob("*.npz")):
            npz = np.load(file_name)
            pos_tensor = torch.from_numpy(npz["pos"])
            neg_tensor = torch.from_numpy(npz["neg"])
          
            # split the sample into half
            half = int(15000 / 2)

            pos_size = pos_tensor.shape[0]
            neg_size = neg_tensor.shape[0]
            
            if pos_size <= half:
                random_pos = (torch.rand(half) * pos_tensor.shape[0]).long()
                sample_pos = torch.index_select(pos_tensor, 0, random_pos)
            else:
                pos_start_ind = random.randint(0, pos_size - half)
                sample_pos = pos_tensor[pos_start_ind : (pos_start_ind + half)]

            if neg_size <= half:
                random_neg = (torch.rand(half) * neg_tensor.shape[0]).long()
                sample_neg = torch.index_select(neg_tensor, 0, random_neg)
            else:
                neg_start_ind = random.randint(0, neg_size - half)
                sample_neg = neg_tensor[neg_start_ind : (neg_start_ind + half)]

            samples = torch.cat([sample_pos, sample_neg], 0)
            
            self.dataset.append(samples)
            
    def __getitem__(self, index):
        return index, self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)