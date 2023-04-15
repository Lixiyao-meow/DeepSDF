import torch
import torch.nn as nn
import torch.nn.functional as F

class Decoder(nn.Module):
    
    def __init__(
        self,
        latent_size=256,
        dims=[512, 512, 512, 512, 512, 512, 512, 512],
        dropout=[0, 1, 2, 3, 4, 5, 6, 7],
        dropout_prob=0.2,
        norm_layers=[0, 1, 2, 3, 4, 5, 6, 7],
        latent_in=[4],
        weight_norm=True):

        super(Decoder, self).__init__()
        
        dims = [latent_size + 3] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

        self.relu = nn.ReLU()
        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    def forward(self, input):
        x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
           
            x = lin(x)
            
            if layer < self.num_layers - 2:
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x