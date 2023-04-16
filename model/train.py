import torch
from torch.utils.data import DataLoader

import numpy as np
import random

from model.dataset import ShapeNet_Dataset
from model.decoder import Decoder

def train_decoder(epochs = 200,
                batch_size=5,
                lat_vecs_std = 0.01,
                decoder_lr = 0.0005,
                lat_vecs_lr = 0.001,
                train_data_path = "./processed_data/train",
                checkpoint_save_path = "./checkpoints/"):

    # ------------ set random seed ------------

    np.random.seed(42)
    torch.manual_seed(42)
    random.seed(42)

    # ------------ setting device on GPU if available, else CPU ------------

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # ------------ load dataset ------------

    train_dataset = ShapeNet_Dataset(train_data_path)
    torch_train = DataLoader(train_dataset,shuffle=True, batch_size=batch_size, num_workers=1)

    # ------------ load auto decoder model ------------
    SDF_autodecoder = Decoder().to(device)

    # ------------ set training parameters ------------

    # initializa latent vectors
    lat_vecs = torch.nn.Embedding(len(train_dataset), 256, max_norm=1.0).cuda()
    torch.nn.init.normal_(lat_vecs.weight.data, 0.0, lat_vecs_std)  #1.0 / np.sqrt(256))

    # set optimizer
    optimizer_all = torch.optim.Adam(
                    [{"params": SDF_autodecoder.parameters(),
                    "lr": decoder_lr},
                    {"params": lat_vecs.parameters(),
                    "lr": lat_vecs_lr}])
    # loss function
    loss_l1 = torch.nn.L1Loss(reduction="sum")

    # other parameters
    minT, maxT = -0.1, 0.1 # clamp

    # ------------ traing process ------------ 

    print("---- start training ----")
    loss_log = []

    for epoch in range(epochs):
        
        SDF_autodecoder.train()

        losses = []
        
        for index, train_data in torch_train:
            
            train_data = train_data.reshape(-1,4).cuda()
            num_sdf_samples = train_data.shape[0]
            train_data.requires_grad = False

            xyz = train_data[:, 0:3]
            sdf_gt = train_data[:, 3].unsqueeze(1)
            sdf_gt = torch.clamp(sdf_gt, minT, maxT)

            optimizer_all.zero_grad()

            # concatenate latent vector and xyz query
            indices = index.cuda().unsqueeze(-1).repeat(1, 15000).view(-1)
            batch_vecs = lat_vecs(indices)
            input = torch.cat([batch_vecs, xyz], dim=1)

            # NN optimization
            pred_sdf = SDF_autodecoder(input)
            pred_sdf = torch.clamp(pred_sdf, minT, maxT)

            loss = loss_l1(pred_sdf, sdf_gt.cuda()) / num_sdf_samples
            loss.backward()
            losses.append(loss.data.mean().cpu())

            optimizer_all.step()
        
        # Print batch loss
        epoch_loss = np.mean(losses)
        loss_log.append(epoch_loss)
        print('[%d/%d] Loss: %.5f' % (epoch+1, epochs, epoch_loss))
        
        torch.save({
                'epoch': epoch,
                'model': SDF_autodecoder.state_dict(),
                'latent_vectors': lat_vecs.state_dict(),
                'optimizer': optimizer_all.state_dict(),
                'loss_log': loss_log,
                }, checkpoint_save_path + str(epoch+1) + ".pt")