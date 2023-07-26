import torch
import torch.nn as nn
import torch.nn.functional as F
from output_padding import *

'''
channels 1->4->8->4->1
'''





class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        # Encoder layers
        self.encoder_conv1 = nn.Conv3d(in_channels=1, out_channels=4, kernel_size=2, stride=2, padding=0)
        self.encoder_conv2 = nn.Conv3d(in_channels=4, out_channels=8, kernel_size=4, stride=4, padding=0)
        self.encoder_fc1 = nn.Linear(262144, 128)
        self.encoder_fc21 = nn.Linear(128, latent_dim)
        self.encoder_fc22 = nn.Linear(128, latent_dim)
        # Decoder layers
        self.decoder_fc3 = nn.Linear(latent_dim, 128)
        self.decoder_fc4 = nn.Linear(128,262144)
        output_padding = output_padding()
        self.decoder_conv3 = nn.ConvTranspose3d(in_channels=8, out_channels=4, kernel_size=4, stride=4, padding=0, output_padding=output_padding)
        self.decoder_conv4 = nn.ConvTranspose3d(in_channels=4, out_channels=1, kernel_size=2, stride=2, padding=0, output_padding=0)

    def encode(self, x):
        print(x.size())
        x = F.relu(self.encoder_conv1(x))
        print(x.size())
        x = F.relu(self.encoder_conv2(x))
        print(x.size())
        x = x.view(x.size(0), -1)
        x = F.relu(self.encoder_fc1(x)) # error here
        mu, logvar = self.encoder_fc21(x), self.encoder_fc22(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z

    def decode(self, z):
        x = F.relu(self.decoder_fc3(z))
        x = F.relu(self.decoder_fc4(x))
        x = x.view(x.size(0), 8, 32, 32, 33)
        x = F.relu(self.decoder_conv3(x))
        x = torch.sigmoid(self.decoder_conv4(x))
        return x

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

