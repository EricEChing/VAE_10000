import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataLoader import train_loader
from vae import *

def vae_loss(recon_batch, data, mu, logvar):
    BCE = F.binary_cross_entropy(recon_batch, data, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    loss = BCE + KLD

    return loss

