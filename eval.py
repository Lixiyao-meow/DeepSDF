"""
This file is for evaluating the prediction of the SDF decoder.
We compare the TP, TN, FP, FN of the prediction with the ground truth.
"""
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder

def decode_sdf(decoder, latent_vector, queries):
    num_samples = queries.shape[0]
    latent_repeat = latent_vector.expand(num_samples, -1)

    inputs = torch.cat([latent_repeat, queries], 1)

    sdf = decoder(inputs)
    
    return sdf

def eval_ROC(gt_sdf, pred_sdf):
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

    return [TP, TN, FP, FN]

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SDF_autodecoder = Decoder().to(device)

# ------------ load dataset ------------

train_data_path = Path("./processed_data/train")
train_dataset = ShapeNet_Dataset(train_data_path)
torch_train = DataLoader(train_dataset,shuffle=True, batch_size=5, num_workers=1)

# ------------ initialization ------------
train_data_size = len(train_dataset)
lat_vecs = torch.nn.Embedding(train_data_size, 256, max_norm=1.0).cuda()

# set optimizer
optimizer_all = torch.optim.Adam(
                [{"params": SDF_autodecoder.parameters(),
                "lr": 0.0005
                },
                {"params": lat_vecs.parameters(),
                "lr": 0.001}])

checkpoint_load_path = "./checkpoints/trained_model.pt"
checkpoint = torch.load(checkpoint_load_path)

last_epoch = checkpoint['epoch']
SDF_autodecoder.load_state_dict(checkpoint['model'])
lat_vecs.load_state_dict(checkpoint['latent_vectors'])
optimizer_all.load_state_dict(checkpoint['optimizer'])
loss_log = checkpoint['loss_log']


# ------------ evaluations ------------

ROC = []

for idx in range(train_data_size):

    latent_code = lat_vecs.weight.data[idx].cuda()
    _, sdf_data = train_dataset[idx]
    queries = sdf_data[:, 0:3].cuda()
    gt_sdf = sdf_data[:, -1]

    pred_sdf = decode_sdf(SDF_autodecoder, latent_code, queries)
    ROC.append(eval_ROC(gt_sdf, pred_sdf))

ROC = np.array(ROC)
#np.save("./ROC.npy", ROC)