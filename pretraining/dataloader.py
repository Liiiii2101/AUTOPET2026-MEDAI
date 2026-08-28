import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
# Dataset Class
class NPZDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.file_names = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.file_paths = [os.path.join(data_dir, f) for f in self.file_names]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        try:
            data = np.load(file_path)['patch']
            image = torch.from_numpy(data).float().squeeze(0)  # Remove the batch dimension
            return image
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            return torch.zeros(2, 128, 128, 128)  # Return a dummy tensor in case of error