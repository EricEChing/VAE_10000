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
input_dim = 512 * 512 * 117  # Total number of elements in the 3D volume
latent_dim = 20  # Choose the desired dimensionality of the latent space
model = VAE(latent_dim)
num_epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_vae(model, train_loader, num_epochs, device)

