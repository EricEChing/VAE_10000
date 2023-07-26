import os
import torch
import torch.utils.data as data
import scipy.io as sio
import torch.nn.functional as F
import math
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PaddedDataset(torch.utils.data.Dataset):

    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = [f for f in os.listdir(data_folder) if f.endswith('.mat')]

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        mat_data = sio.loadmat(os.path.join(self.data_folder, mat_file))
        volume = mat_data['T00_new']  # Assuming the data is stored under the key 'T00'
        # Normalize the data to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        # Convert to torch tensor and add a channel dimension (assuming grayscale images)
        volume = torch.unsqueeze(torch.tensor(volume),0)
        volume = volume.to(dtype=torch.float)
        print(volume.size())
        if(index == 1):
            volume = F.pad(volume,(0,15))

        return volume


    def __len__(self):
        return len(self.file_list)
'''
def collate_fn(batch):
    # Find the maximum depth in the current batch

    # Apply padding to each volume separately
    padded_batch = []
    
    padded_batch.append(item)

    return torch.stack(padded_batch)
'''

from torch.utils.data import DataLoader

# Assuming you have a folder 'data_folder' containing the .mat files
data_folder = '/home/gabriela/PycharmProjects/VAE_10000/inputMATfiles'


# Create the MATPaddedDataset and DataLoader
batch_size = 1
dataset = PaddedDataset(data_folder)

train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

