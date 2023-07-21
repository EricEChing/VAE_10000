import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataLoader import train_loader
from train_vae import *
from vae import *
def vae_loss(recon_x, x, mu, logvar):
    # loss function MODIFIER LEARNME
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL divergence regularization LEARNME
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD
