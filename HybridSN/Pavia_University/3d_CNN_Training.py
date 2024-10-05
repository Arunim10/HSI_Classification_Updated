import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from tqdm import tqdm
import random
from scipy.io import loadmat
import albumentations as A
from torch.utils.data import DataLoader

from pavia_data import train_test_split_data
from Build_Dataset_Class import PaviaU_Dataset
from HybridSN_Model import HybridCNN
from utils import f_score,accuracy,evaluate_model,gc_collect,train_model


def train():
    ## LOADING DATA
    paviaU = loadmat(r'D:\HSI Project\Updated_Work\HSI_Classification\HybridSN\PaviaU.mat')['paviaU']
    paviaU_gt = loadmat(r'D:\HSI Project\Updated_Work\HSI_Classification\HybridSN\PaviaU_gt.mat')['paviaU_gt']

    (train_X,train_Y),(valid_X,valid_Y) = train_test_split_data(kernel_size=25,data=paviaU,data_gt=paviaU_gt,num_components=15)

    ## Build Dataset


    # data_transforms = {
    #     "train": A.Compose([
    # #         A.HorizontalFlip(p=0.5),
    # #         A.VerticalFlip(p=0.5),
    #         A.Normalize(mean=1389.1253099399873, 
    #                     std=897.6575399774091)
    #         ])
    #         ,
        
    #     "valid": A.Compose([
    #         A.Normalize(mean=1389.1253099399873, 
    #                     std=897.6575399774091)
    #         ])
    # }
    

    train_data = PaviaU_Dataset(train_X,train_Y,transforms=None)
    test_data = PaviaU_Dataset(valid_X,valid_Y,transforms=None)
    train_dl = DataLoader(train_data,batch_size=256,num_workers=4,shuffle=True,pin_memory=True,drop_last=False)
    valid_dl = DataLoader(test_data,batch_size=128,num_workers=4,shuffle=False,pin_memory=True)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device active: ",device)
    gc_collect()

    model = HybridCNN(15,25,9)
    model = model.to(device)

    ## Training
    epoch = 50
    model,train_loss,val_loss,train_fscore,val_fscore = train_model(model,train_dl,valid_dl,epoch,device,1e-3,1e-4)
    
    fig,ax = plt.subplots(1,2,figsize=(10,10))
    epochs = range(1,epoch+1)
    # Plot train and validation losses
    ax[0].plot(epochs, train_loss, label='Train Loss',color='blue')
    ax[0].plot(epochs, val_loss, label='Validation Loss',color='orange')

    # Add labels and title
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Loss')
    ax[0].set_title('Train and Validation Losses')

    # Add legend
    ax[0].legend()


    # Train and Val F-Scores
    ax[1].plot(epochs, train_fscore, label='Train F-Score',color='blue')
    ax[1].plot(epochs, val_fscore, label='Validation F-Score',color='orange')

    # Add labels and title
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('F-Score')
    ax[1].set_title('Train and Validation F-Scores')

    # Add legend
    ax[1].legend()

    # Show plot
    plt.show()
        
    return model

if __name__ == '__main__':
    model = train()