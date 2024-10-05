import torch
from scipy.io import loadmat
import numpy as np

# paviaU_gt = loadmat(r'D:\HSI Project\Updated_Work\Paper_Model_Building_Pytorch_Notebooks\CNN3D\PaviaU_gt.mat')['paviaU_gt']
# target_list = list(np.unique(paviaU_gt))

# target_list = []
# for i in range(1,10):
#     target_list.append(f'label_{i}')
# # print(target_list)
# mapping = {}
# for i in range(len(target_list)):
#     mapping[target_list[i]] = i
# # print(mapping)
# class PaviaU_Dataset(torch.utils.data.Dataset):
#     def __init__(self,data,labels,transforms=None):
#         super().__init__()
#         self.data = data
#         self.labels = labels
#         self.transforms = transforms
    
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self,idx):
#         label = self.labels[idx]
#         spectral_values = torch.tensor(self.data[idx].squeeze(0)).permute(1,2,0).numpy()
#         if self.transforms is not None:
#             data = self.transforms(image=spectral_values)
#             spectral_values = data['image']
#         return torch.tensor(spectral_values).permute(2,0,1).unsqueeze(0),torch.tensor(mapping[label])

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