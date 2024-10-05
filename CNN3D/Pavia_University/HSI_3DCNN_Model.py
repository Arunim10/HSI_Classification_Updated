import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
# class HSI_3DCNN(nn.Module):
#     def __init__(self):
#         super(HSI_3DCNN, self).__init__()
#         # Define convolutional layers
#         self.model = nn.Sequential(
#                         nn.Conv3d(in_channels=1, out_channels=20, kernel_size=(3, 3, 3), padding=(0, 0, 1)),
#                         nn.BatchNorm3d(20),
#                         nn.ReLU(inplace=True),
#                         nn.Conv3d(in_channels=20, out_channels=2, kernel_size=(1, 1, 3), stride=(1, 1, 2), padding=(0, 0, 1)),
#                         nn.BatchNorm3d(2),
#                         nn.ReLU(inplace=True),
#                         # nn.Dropout(p=0.1),
#                         nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
#                         nn.BatchNorm3d(35),
#                         nn.ReLU(inplace=True),
#                         nn.Conv3d(in_channels=35, out_channels=2, kernel_size=(1, 1, 2), stride=(1, 1, 2), padding=(0, 0, 1)),
#                         # nn.BatchNorm3d(2),
#                         nn.ReLU(inplace=True),
#                         # nn.Dropout(p=0.1),
#                         nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
#                         nn.BatchNorm3d(35),
#                         nn.ReLU(inplace=True),
#                         nn.Conv3d(in_channels=35, out_channels=2, kernel_size=(1, 1, 1), stride=(1, 1, 2), padding=(0, 0, 1)),
#                         nn.BatchNorm3d(2),
#                         nn.ReLU(inplace=True),
#                         # nn.Dropout(p=0.1),
#                         nn.Conv3d(in_channels=2, out_channels=35, kernel_size=(1, 1, 3), padding=(0, 0, 1)),
#                         nn.BatchNorm3d(35),
#                         nn.ReLU(inplace=True),
#                         nn.Conv3d(in_channels=35, out_channels=4, kernel_size=(1, 1, 1), stride=(2, 2, 2), padding=(0, 0, 0)),
#                         nn.BatchNorm3d(4),
#                         nn.ReLU(inplace=True),
#                         # nn.Dropout(p=0.1)
#                     )
        
        
#         x = torch.rand(1,1,103,11,11)
#         num_channels = self.model(x).view(1,-1).shape[1]
#         # Define fully-connected layer
#         self.fc9 = nn.Linear(in_features=num_channels, out_features=9)

#         # Initialize weights and biases (using MSRA initializer for weights and constant zero for biases)
#         for m in self.modules():
#             if isinstance(m, nn.Conv3d):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(x.size(0), -1)  # Flatten the output
#         x = F.softmax(self.fc9(x),dim=-1)
#         return x


class HSI_3DCNN(nn.Module):
    def __init__(self,num_classes):
        super(HSI_3DCNN, self).__init__()
        self.model = nn.Sequential(
        nn.Conv3d(in_channels=1, out_channels=32, kernel_size=(24, 5, 5)),
        nn.BatchNorm3d(32),
        nn.ReLU(inplace=True),
        nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(16, 5, 5)),
        nn.BatchNorm3d(64),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(kernel_size=(1, 2, 2))
        )
        x = torch.rand(1,1,50,11,11)
        num_channels = self.model(x).view(1,-1).shape[1]
        self.fc1 = nn.Linear(num_channels, 300)
        self.bn3 = nn.BatchNorm1d(300)
        self.fc2 = nn.Linear(300, num_classes)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)  # Flatten the output
        x = self.fc1(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
    
