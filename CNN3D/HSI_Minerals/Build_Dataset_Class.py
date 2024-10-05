import pandas as pd
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from utils import padWithZeros,createImageCubes,fill_nan
import torch
import torch.nn.functional as F
from glob import glob

class Minerals_Dataset(torch.utils.data.Dataset):
    def __init__(self,df,transforms=None):
        super().__init__()
        self.df = df
        self.transforms = transforms
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self,idx):
        
        patch_path = self.df.iloc[idx]['Patch_Path']
        patch = np.load(patch_path) # 5x5x256
        label = self.df.iloc[idx]['Label']
        
        # Handling NaN Values
        ind = np.where(np.isnan(patch))
        for (x,y,c) in zip(ind[0],ind[1],ind[2]):
            patch = fill_nan(patch,(x,y,c))  
        
        patch = np.transpose(F.pad(torch.tensor(np.transpose(patch,axes=(2,0,1))),(1,1,1,1)).numpy(),axes=(1,2,0))
        
        if self.transforms is not None:
            data = self.transforms(image=patch)
            patch = data['image']
        return torch.tensor(patch).permute(2,0,1).unsqueeze(0),torch.tensor(label)
    
        
        
        
        
        