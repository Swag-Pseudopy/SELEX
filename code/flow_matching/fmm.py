import torch
from torch.utils.data import Dataset, DataLoader
import glob
import os
import numpy as np

from functools import partial
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
from zuko.utils import odeint
# from torchdiffeq import odeint
# import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Net(nn.Module):
    def __init__(self, in_dim=2, out_dim=2, h_dims=[512]*5, n_freqs=10, round_embedding_dim=4):
        super().__init__()
        self.round_embedding_dim = round_embedding_dim
        ins = [in_dim + 2 * n_freqs + round_embedding_dim] + h_dims
        outs = h_dims + [out_dim]
        self.n_freqs = n_freqs

        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(in_d, out_d), nn.LeakyReLU()) for in_d, out_d in zip(ins, outs)
        ])
        self.top = nn.Linear(out_dim, out_dim)
        self.round_embedding = nn.Embedding(1000, round_embedding_dim)  # Adjust max rounds if needed

    def time_encoder(self, t):
        freq = 2 * torch.arange(self.n_freqs, device=t.device) * torch.pi
        t = freq * t[..., None]
        return torch.cat((t.cos(), t.sin()), dim=-1)

    def forward(self, t, x, round_num):
        t_encoded = self.time_encoder(t)
        if t_encoded.dim() == 2:
            t_encoded = t_encoded.unsqueeze(1).expand(-1, x.shape[1], -1) if x.dim() == 3 else t_encoded
        round_embed = self.round_embedding(round_num)
        if round_embed.dim() == 2 and x.dim() == 3:
            round_embed = round_embed.unsqueeze(1).expand(-1, x.shape[1], -1)
        # print(t_encoded.shape, round_embed.shape, x.shape);exit(0)
        x = torch.cat((x, t_encoded, round_embed), dim=-1)
        for layer in self.layers:
            x = layer(x)
        return self.top(x)

class CondVF(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, t, x, round_num):
        return self.net(t, x, round_num)

    def wrapper(self, t, x, round_num):
        t = t * torch.ones(len(x), device=x.device)
        return self(t, x, round_num)

    def decode(self, x_0, round_num, t0=0., t1=1.):
        wrapped_func = partial(self.wrapper, round_num=round_num)
        return odeint(wrapped_func, x_0, 0., 1., self.parameters())

class OTFlowMatching:
    def __init__(self, sig_min=0.001):
        self.sig_min = sig_min
        self.eps = 1e-5

    def psi_t(self, x, x_1, t):
        # print(t.shape, x.shape, x_1.shape)
        t = t[..., None].expand(x.shape)
        # print(t.shape, x.shape, x_1.shape)
        # exit(0)
        return (1 - (1 - self.sig_min) * t) * x + t * x_1

    def loss(self, v_t, x_0, x_1, round_num):
        t = (torch.rand(x_0.size(0), device=x_1.device) + torch.arange(x_0.size(0), device=x_1.device) / len(x_1)) % (1 - self.eps)
        t = t[:, None].expand(x_1.shape[0], x_1.shape[1])
        # x_0 = torch.randn_like(x_1)
        v_psi = v_t(t[:, 0], self.psi_t(x_0, x_1, t), round_num)
        d_psi = x_1 - (1 - self.sig_min) * x_0
        return torch.mean((v_psi - d_psi) ** 2)

# Generate synthetic data for 1000 sequences, each with 2 features
# n_samples = 1024
# n_features = 2
# data = np.random.rand(n_samples, n_features).astype(np.float32)

# # Create a TensorDataset and DataLoader
# dataset = TensorDataset(torch.from_numpy(data))  # No labels here
# batch_size = 64
# dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# # Initialize the network, conditional velocity field, and OTFlowMatching model
# net = Net(in_dim=2, out_dim=2, h_dims=[512]*5, n_freqs=10).to(device)
# v_t = CondVF(net).to(device)
# model = OTFlowMatching(sig_min=0.001)

# # Optimizer for the network
# optimizer = torch.optim.Adam(v_t.parameters(), lr=1e-3)

# num_epochs = 100  # Number of epochs to train

# for epoch in range(num_epochs):
#     for batch in dataloader:
#         # x_0 from the dataloader
#         x_0 = batch[0].to(device)

#         # Sample random round `r` for each batch
#         round_num = torch.randint(0, 1000, (batch_size,), device=device)  # Random round selection (adjust as needed)

#         # Sample random x_1 (next batch of the same size or some other random data)
#         x_1 = torch.randn_like(x_0)  # Randomly sampled target

#         # Compute the loss for the entire batch in parallel
#         loss = model.loss(v_t, x_0, x_1, round_num)

#         # Backpropagation
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#     print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}",end = "\n") if not (epoch+1)%10 else print(end = "")
