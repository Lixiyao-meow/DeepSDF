import torch

import numpy as np

from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder
import model.reconstruct as reconstruct

# ------------ setting device on GPU if available, else CPU ------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ------------ load example test sample ------------
val_data_path = "./processed_data/validation"
val_dataset = ShapeNet_Dataset(val_data_path)

# ------------ load decoder ------------
decoder = Decoder().to(device)
checkpoint = torch.load("./checkpoints/200.pt")
decoder.load_state_dict(checkpoint["model"])

for idx in range(2, 15):
    
    test_sample = val_dataset[idx]

    # ------------ reconstruction from a validation sample ------------
    filename = "./reconstruction/example_" + str(51+idx)
    reconstruct.reconstruct(test_sample,
                            decoder,
                            filename,
                            lat_iteration=800,
                            lat_init_std = 0.01, 
                            lat_lr = 5e-4,
                            N=256, 
                            max_batch=32 ** 3)