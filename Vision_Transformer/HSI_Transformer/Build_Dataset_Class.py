import pandas as pd
from scipy.io import loadmat
import numpy as np
from tqdm import tqdm
from utils import padWithZeros,createImageCubes,fill_nan,patch_from_hsi
import torch
import torch.nn.functional as F
from glob import glob
from functools import lru_cache
# import nvidia

# global cache

# cache = {}

class Minerals_Dataset(torch.utils.data.Dataset):
    def __init__(self,df,mmaps,window_size,transforms=None):
        super().__init__()
        self.df = df
        self.window_size = window_size
        self.transforms = transforms
        self.mmaps = mmaps
    def __len__(self):
        return len(self.df)
    
    # @lru_cache(maxsize=128)
    def patch_from_hsi(self,id,hsi_path,window_size):
        # hsi = np.transpose(loadmat(hsi_path)['HDR'],axes=(1,2,0))
        hsi = self.mmaps[hsi_path]
        # hsi = self.hsi_dict[hsi_path]
        row,col,channels = hsi.shape
        x_hsi,y_hsi = id//col,id%col
        margin = int((window_size-1)/2)
        
        patch = np.zeros((window_size,window_size,channels))
        
        lower_bound = [max(0,x_hsi-margin),max(0,y_hsi-margin)]
        upper_bound = [min(row-1,x_hsi+margin),min(col-1,y_hsi+margin)]
        
        diff_lower = [abs(x_hsi-lower_bound[0]),abs(y_hsi-lower_bound[1])]
        diff_upper = [abs(upper_bound[0]-x_hsi),abs(upper_bound[1]-y_hsi)]
        
        patch_lower_bound = [margin-diff_lower[0],margin-diff_lower[1]]
        patch_upper_bound = [margin+diff_upper[0],margin+diff_upper[1]]
        
        patch[patch_lower_bound[0]:patch_upper_bound[0]+1,patch_lower_bound[1]:patch_upper_bound[1]+1,:] = hsi[lower_bound[0]:upper_bound[0]+1,lower_bound[1]:upper_bound[1]+1,:]
        # del hsi
        return patch
    
    def __getitem__(self,idx):
        
        hsi_path = self.df.iloc[idx]['hsi_path']
        Id = self.df.iloc[idx]['ID']
        exp = self.df.iloc[idx]['EXP']
        patch_id = self.df.iloc[idx]['patch_id']
        label = self.df.iloc[idx]['patch_Label']
        
        patch = self.patch_from_hsi(patch_id,hsi_path,window_size=self.window_size)
        
        # Handling NaN Values
        # ind = np.where(np.isnan(patch))
        # for (x,y,c) in zip(ind[0],ind[1],ind[2]):
        #     patch = fill_nan(patch,(x,y,c))  
            
        # patch = np.transpose(F.pad(torch.tensor(np.transpose(patch,axes=(2,0,1))),(1,1,1,1)).numpy(),axes=(1,2,0))
        
        if self.transforms is not None:
            data = self.transforms(image=patch)
            patch = data['image']
        return torch.tensor(patch).permute(2,0,1),torch.tensor(label)
    
    
# class Minerals_Dataset(torch.utils.data.Dataset):
#     def __init__(self,df,transforms=None):
#         super().__init__()
#         self.df = df
#         self.transforms = transforms
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self,idx):
        
#         patch_path = self.df.iloc[idx]['Patch_Path']
#         patch = np.load(patch_path)
#         label = self.df.iloc[idx]['Label']
        
#         # Handling NaN Values
#         # ind = np.where(np.isnan(patch))
#         # for (x,y,c) in zip(ind[0],ind[1],ind[2]):
#         #     patch = fill_nan(patch,(x,y,c))  
            
#         patch = np.transpose(F.pad(torch.tensor(np.transpose(patch,axes=(2,0,1))),(1,1,1,1)).numpy(),axes=(1,2,0))
        
#         if self.transforms is not None:
#             data = self.transforms(image=patch)
#             patch = data['image']
#         return torch.tensor(patch).permute(2,0,1),torch.tensor(label)
        
        
        
        
        