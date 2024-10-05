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
        patch = np.load(patch_path)
        label = self.df.iloc[idx]['Label']
        
        # Handling NaN Values
        # ind = np.where(np.isnan(patch))
        # for (x,y,c) in zip(ind[0],ind[1],ind[2]):
        #     patch = fill_nan(patch,(x,y,c))  
            
        patch = np.transpose(F.pad(torch.tensor(np.transpose(patch,axes=(2,0,1))),(1,1,1,1)).numpy(),axes=(1,2,0))
        
        if self.transforms is not None:
            data = self.transforms(image=patch)
            patch = data['image']
        return torch.tensor(patch).permute(2,0,1).unsqueeze(0),torch.tensor(label)
    
    
# import pandas as pd
# from scipy.io import loadmat
# import numpy as np
# from tqdm import tqdm
# from utils import padWithZeros,createImageCubes,fill_nan,patch_from_hsi
# import torch
# import torch.nn.functional as F
# from glob import glob

# class Minerals_Dataset(torch.utils.data.Dataset):
#     def __init__(self,df,window_size,transforms=None):
#         super().__init__()
#         self.df = df
#         self.window_size = window_size
#         self.transforms = transforms
#     def __len__(self):
#         return len(self.df)
    
#     def __getitem__(self,idx):
        
#         hsi_path = self.df.iloc[idx]['hsi_path']
#         Id = self.df.iloc[idx]['ID']
#         exp = self.df.iloc[idx]['EXP']
#         patch_id = self.df.iloc[idx]['patch_id']
#         label = self.df.iloc[idx]['patch_Label']
        
#         patch = patch_from_hsi(patch_id,hsi_path,window_size=self.window_size)
        
#         # Handling NaN Values
#         # ind = np.where(np.isnan(patch))
#         # for (x,y,c) in zip(ind[0],ind[1],ind[2]):
#         #     patch = fill_nan(patch,(x,y,c))  
            
#         # patch = np.transpose(F.pad(torch.tensor(np.transpose(patch,axes=(2,0,1))),(1,1,1,1)).numpy(),axes=(1,2,0))
        
#         if self.transforms is not None:
#             data = self.transforms(image=patch)
#             patch = data['image']
#         return torch.tensor(patch).permute(2,0,1).unsqueeze(0),torch.tensor(label)
    
        
        
        
        
        
        
        
        
        
        