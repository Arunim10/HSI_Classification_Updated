import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

class HSI_3DCNN(nn.Module):
    def __init__(self, in_chs, patch_size,class_nums):
        super(HSI_3DCNN,self).__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        self.conv3dmodel = nn.Sequential(
                            nn.Conv3d(in_channels=1,out_channels=4,kernel_size=(8, 3, 3)),
                            nn.BatchNorm3d(4),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(in_channels=4,out_channels=8,kernel_size=(4, 3, 3)),
                            nn.BatchNorm3d(8),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.3),
                            nn.MaxPool3d(kernel_size=(4,2,2))
                            )
        temp = self.conv3dmodel(torch.rand(2,1,in_chs,patch_size,patch_size))
        self.dim3d = temp.view(2,-1).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(self.dim3d,256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256,class_nums)
        )

    def forward(self,X):
        x = self.conv3dmodel(X)
        x = x.view(x.shape[0],-1)
        x = self.linear(x)
        return x
    


