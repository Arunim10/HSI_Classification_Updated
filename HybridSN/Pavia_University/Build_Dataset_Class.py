import torch
from scipy.io import loadmat
import numpy as np

class PaviaU_Dataset(torch.utils.data.Dataset):
    def __init__(self,data,labels,transforms=None):
        super().__init__()
        self.data = data
        self.labels = labels
        self.transforms = transforms
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        label = self.labels[idx]
        spectral_values = self.data[idx]
        if self.transforms is not None:
            data = self.transforms(image=spectral_values)
            spectral_values = data['image']
        return torch.tensor(spectral_values).permute(2,0,1).unsqueeze(0),torch.tensor(label)