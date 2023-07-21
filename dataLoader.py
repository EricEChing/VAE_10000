import os
import torch
import torch.utils.data as data
import scipy.io as sio
import torch.nn.functional as F

class MATPaddedDataset(data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = [f for f in os.listdir(data_folder) if f.endswith('.mat')]

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        mat_data = sio.loadmat(os.path.join(self.data_folder, mat_file))
        volume = mat_data['T00']  # Assuming the data is stored under the key 'T00'

        # Normalize the data to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        # Convert to torch tensor and add a channel dimension (assuming grayscale images)
        volume = torch.tensor(volume).unsqueeze(0).float()

        return volume

    def __len__(self):
        return len(self.file_list)

def collate_fn(batch):
    # Find the maximum depth in the current batch
    max_depth = max([item.size(1) for item in batch])

    # Apply padding to each volume separately
    padded_batch = []
    for item in batch:
        depth_padding = max_depth - item.size(1)
        padded_item = F.pad(item, (0, 0, 0, 0, 0, depth_padding))
        padded_batch.append(padded_item)

    return torch.stack(padded_batch)

# Assuming you have a folder 'data_folder' containing the .mat files
data_folder = 'path/to/your/folder'


import torch
from torch.utils.data import DataLoader

# Assuming you have a folder 'data_folder' containing the .mat files
data_folder = '/home/gabriela/PycharmProjects/VAE_10000/inputMATfiles'

# Assuming you already know the maximum depth of all volumes in the dataset

# Create the MATPaddedDataset and DataLoader
batch_size = 2
dataset = MATPaddedDataset(data_folder)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4,collate_fn=collate_fn)

