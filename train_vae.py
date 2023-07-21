import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dataLoader import train_loader
from vae_loss import *
from vae import *
# Training loop
def train_vae(model, train_loader, num_epochs, device):
    model.to(device)
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(1, num_epochs + 1):
        train_loss = 0

        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)  # If needed, data is in tensor form here
            optimizer.zero_grad()

            recon_batch, mu, logvar = model(data)
            loss = vae_loss(recon_batch, data, mu, logvar)

            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print('Epoch: {} Average Loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))


    print('Epoch: {} Train Loss: {:.4f}'.format(epoch, train_loss / len(train_loader.dataset)))
