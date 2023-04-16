from pathlib import Path
import torch
from torch.utils.data import DataLoader

from dataset import ShapeNet_Dataset
from decoder import Decoder

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SDF_autodecoder = Decoder().to(device)

# ------------ initialization ------------
train_data_size = 50
lat_vecs = torch.nn.Embedding(train_data_size, 256, max_norm=1.0).cuda()
torch.nn.init.normal_(lat_vecs.weight.data, 0.0, 0.01)  #1.0 / np.sqrt(256))

# set optimizer
optimizer_all = torch.optim.Adam(
                [{"params": SDF_autodecoder.parameters(),
                "lr": 0.0005
                },
                {"params": lat_vecs.parameters(),
                "lr": 0.001}])

checkpoint_load_path = "./checkpoints/200.pt"
checkpoint = torch.load(checkpoint_load_path)

last_epoch = checkpoint['epoch']
SDF_autodecoder.load_state_dict(checkpoint['model'])
lat_vecs.load_state_dict(checkpoint['latent_vectors'])
optimizer_all.load_state_dict(checkpoint['optimizer'])
loss_log = checkpoint['loss_log']

# ------------ load dataset ------------

train_data_path = Path("./processed_data/train")
val_data_path = Path("./processed_data/validation/")
train_dataset = ShapeNet_Dataset(train_data_path)
val_dataset = ShapeNet_Dataset(val_data_path)

torch_train = DataLoader(train_dataset,shuffle=True, batch_size=5, num_workers=1)
torch_validation = DataLoader(val_dataset,shuffle=True, batch_size=5, num_workers=1)

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)

    inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)
    
    return sdf

idx = 0
latent_code = lat_vecs.weight.data[idx].cuda()
_, sdf_data = train_dataset[idx]
queries = sdf_data[:, 0:3].cuda()
gt_sdf = sdf_data[:, -1]

pred_sdf = decode_sdf(SDF_autodecoder, latent_code, queries)

TP, TN, FP, FN = 0, 0, 0, 0
n = gt_sdf.shape[0]

for i in range(n):
    if gt_sdf[i] > 0 and pred_sdf[i] > 0:
        TP += 1
    elif gt_sdf[i] > 0 and pred_sdf[i] < 0:
        FN += 1
    elif gt_sdf[i] < 0 and pred_sdf[i] < 0:
        TN += 1
    else:
        FP += 1

print(TP, TN, FP, FN)