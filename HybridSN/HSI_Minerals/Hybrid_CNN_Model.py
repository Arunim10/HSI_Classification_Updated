import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F  
from torchcrf import CRF

class HybridCNN(nn.Module):
    def __init__(self, in_chs, patch_size,C1,C3, class_nums):
        super(HybridCNN,self).__init__()
        self.in_chs = in_chs
        self.patch_size = patch_size
        self.conv3dmodel = nn.Sequential(
                            nn.Conv3d(in_channels=1,out_channels=8,kernel_size=(8, 3, 3)),
                            nn.BatchNorm3d(8),
                            nn.ReLU(inplace=True),
                            nn.Conv3d(in_channels=8,out_channels=8,kernel_size=(4, 3, 3)),
                            nn.BatchNorm3d(8),
                            nn.ReLU(inplace=True),
                            nn.Dropout(0.3)
                            )
        temp = self.conv3dmodel(torch.rand(2,1,in_chs,patch_size,patch_size))
        self.dim3d = temp.shape[1] * temp.shape[2]
        
        self.conv2dmodel = nn.Sequential(
            nn.Conv2d(self.dim3d,C1,kernel_size=3),
            nn.BatchNorm2d(C1),
            nn.ReLU(),
            nn.Dropout(0.3),
            # nn.Conv2d(C1,C3,kernel_size=1),
            # nn.BatchNorm2d(C3),
            # nn.ReLU(),
            # nn.Dropout(0.3),
        )
        
        temp = temp.view(temp.shape[0],temp.shape[1]*temp.shape[2],temp.shape[3],temp.shape[4])
        temp = self.conv2dmodel(temp)
        self.dim2d = temp.view(temp.shape[0],-1).shape[1]
        self.linear = nn.Sequential(
            nn.Linear(self.dim2d,64),
            nn.ReLU(),
            nn.Linear(64,class_nums)
        )
        
    def forward(self,X):
        x = self.conv3dmodel(X)
        x = x.view(x.shape[0],x.shape[1]*x.shape[2],x.shape[3],x.shape[4])
        x = self.conv2dmodel(x)
        x = x.contiguous().view(x.shape[0],-1)
        x = self.linear(x)
        # x = self.crf(x)
        return x
    
    

