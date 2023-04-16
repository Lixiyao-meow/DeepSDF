import torch

from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder
import model.reconstruct as reconstruct

# ------------ setting device on GPU if available, else CPU ------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

# ------------ load validation samples ------------
val_data_path = "./processed_data/validation"
val_dataset = ShapeNet_Dataset(val_data_path)

# ------------ load decoder ------------
decoder = Decoder().to(device)
checkpoint = torch.load("./checkpoints/trained_model.pt")
decoder.load_state_dict(checkpoint["model"])

# ------------ reconstruction ------------
for idx in range(len(val_dataset)):
    
    test_sample = val_dataset[idx]

    filename = "./reconstruction/example_" + str(51+idx)
    reconstruct.reconstruct(test_sample,
                            decoder,
                            filename,
                            lat_iteration=800,
                            lat_init_std = 0.01, 
                            lat_lr = 5e-4,
                            N=256, 
                            max_batch=32 ** 3)