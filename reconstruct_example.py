import torch

import numpy as np

from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder
import model.reconstruct as reconstruct

# load example test sample
val_data_path = "./processed_data/validation"
val_dataset = ShapeNet_Dataset(val_data_path)

idx = np.random.randint(len(val_dataset))
test_sample = val_dataset[idx]


# load decoder
decoder = Decoder()
checkpoint = torch.load("./checkpoints/200.pt")
decoder.load_state_dict(checkpoint["model"])

# reconstruction from a validation sample
filename = "./reconstruction/example_" + str(idx) + ".obj"
reconstruct.reconstruct(test_sample,
                        decoder,
                        filename,
                        lat_iteration=20,
                        lat_init_std = 0.01, 
                        lat_lr = 5e-4,
                        N=128, 
                        max_batch=16 ** 3)