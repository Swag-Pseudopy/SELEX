import os
import sys
import time
import glob
import datetime
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
# from IPython.display import clear_output, display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

class CustomWeightedLoss(nn.Module):
    """KL Divergence Loss"""
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        global rho_current, sample_indices
        rho_current = rho_current.to(y_pred.device).detach()
        sample_indices = sample_indices.to(y_pred.device).detach()

        # # Debug check
        # assert y_pred.requires_grad, "y_pred does not require grad!"

        weights = F.normalize(rho_current[sample_indices].detach(), p=1, dim=0)
        diff = torch.log(F.normalize(torch.exp(weights-y_true.float()),p=1,dim=0)/F.normalize(torch.exp(weights-y_pred.float()),p=1,dim=0))
        weighted_diff = weights * diff
        
        # # Debug checks
        # print("y_pred.requires_grad:", y_pred.requires_grad)
        # print("y_pred.grad_fn:", y_pred.grad_fn)
        # print("y_true.requires_grad:", y_true.requires_grad)
        # print("y_true.grad_fn:", y_true.grad_fn)
        # print("rho_current.requires_grad:", rho_current.requires_grad)
        # print("rho_current.grad_fn:", rho_current.grad_fn if hasattr(rho_current, 'grad_fn') else None)
        # print("sample_indices.requires_grad:", sample_indices.requires_grad)
        # print("sample_indices.grad_fn:", sample_indices.grad_fn if hasattr(sample_indices, 'grad_fn') else None)
        # print("weights.requires_grad:", weights.requires_grad)
        # print("weights.grad_fn:", weights.grad_fn if hasattr(weights, 'grad_fn') else None)
        # print("diff.requires_grad:", diff.requires_grad)
        # print("diff.grad_fn:", diff.grad_fn if hasattr(diff, 'grad_fn') else None)
        # print("weighted_diff.requires_grad:", weighted_diff.requires_grad)
        # print("weighted_diff.grad_fn:", weighted_diff.grad_fn if hasattr(weighted_diff, 'grad_fn') else None)
        print(weighted_diff.shape)
        # exit(0)
        
        return torch.mean(weighted_diff)

model = nn.Linear(10, 1)
criterion = CustomWeightedLoss()
x = torch.randn(5, 10, requires_grad=True)
y_true = torch.randn(5, 1)
rho_current = torch.ones(5, requires_grad=False)
sample_indices = torch.tensor([0, 1, 2, 3, 4])

y_pred = model(x)
loss = criterion(y_pred, y_true)
print("Loss:", loss.item())
print("Loss requires_grad:", loss.requires_grad)  # Must be True
print("Loss grad_fn:", loss.grad_fn)  # Must NOT be None
loss.backward()  # This must work
