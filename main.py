# VAE intended for unsupervised learning of nifti/mat files of COPD cases
# user discretion is advised

# assume dataset is in MAT files (256x256xDEPTH) or (512x512xDEPTH)
# figure out what 589824 is
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataLoader import train_loader
from vae_loss import *
from train_vae import *
from vae import *

# Train the VAE for multiple epochs
latent_dim = 64  # Fix when memory issues get fixed
model = VAE(latent_dim)
num_epochs = 10
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
torch.cuda.set_per_process_memory_fraction(1., 0)
model = model.to(device)
train_vae(model, train_loader, num_epochs, device)

