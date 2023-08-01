import os
import torch
import torch.utils.data as data
import scipy.io as sio
import torch.nn.functional as F
from torch.utils.data import DataLoader

class MATPaddedDataset(data.Dataset):
    def __init__(self, data_folder):
        self.data_folder = data_folder
        self.file_list = [f for f in os.listdir(data_folder) if f.endswith('.mat')]

    def __getitem__(self, index):
        mat_file = self.file_list[index]
        mat_data = sio.loadmat(os.path.join(self.data_folder, mat_file))
        volume = mat_data['new_T00']  # Assuming the data is stored under the key 'T00'

        # Normalize the data to [0, 1]
        volume = (volume - volume.min()) / (volume.max() - volume.min())

        # Convert to torch tensor and add a channel dimension (assuming grayscale images)
        volume = torch.tensor(volume).unsqueeze(0).float()
        return volume

    def __len__(self):
        return len(self.file_list)




# Assuming you have a folder 'data_folder' containing the .mat files
data_folder = 'C:\\Users\ericc\PycharmProjects\VAE_10000\inputMATfiles'
test_folder = 'C:\\Users\ericc\PycharmProjects\VAE_10000\\testMATfiles'

# Assuming you already know the maximum depth of all volumes in the dataset

# Create the MATPaddedDataset and DataLoader
batch_size = 3
dataset = MATPaddedDataset(data_folder)
testset = MATPaddedDataset(test_folder)
train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
valid_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=0)
