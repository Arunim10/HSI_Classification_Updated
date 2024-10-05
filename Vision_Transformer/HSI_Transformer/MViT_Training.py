import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
from tqdm import tqdm
import random
from scipy.io import loadmat
import albumentations as A
from torch.utils.data import DataLoader
import time
from glob import glob
import random
import os 
import warnings
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import torch.multiprocessing as mp
import pickle

from Build_Dataset_Class import Minerals_Dataset
from MViT_Model import MViT
from utils import *



def train():
    # Suppress all warnings
    warnings.filterwarnings("ignore")
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'   
    ## LOADING DATA
    patch_df = pd.read_csv("E:\\Patches_Data\\Patch_df_for_any_window_size.csv")
    # print(patch_df.iloc[0]['hsi_path'])
    
    # df = pd.read_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_Mapped.csv")
    # print(df.iloc[0]['HSI_Path'])
    # patch_df = patch_df.sample(frac=1).reset_index(drop=True)
    # *************************************** UPSAMPLING AND DOWNSAMPLING ****************************************
    
    # Identify the class distribution
    class_counts = patch_df['patch_Label'].value_counts()
    print("Original class distribution:\n", class_counts)

    # # Determine the majority class and the number of samples in the majority class
    majority_class = class_counts.idxmax()
    majority_count = class_counts.iloc[1]
    print("majority_class: ",majority_class)
    print("majority_count: ",majority_count)

    # # Create a list to hold the resampled DataFrames
    resampled_data = []

    print("Upsampling minority")
    # Upsample minority classes
    for label in tqdm(class_counts.index):
        if label!=majority_class:
            class_subset = patch_df[patch_df['patch_Label'] == label]
            resampled_class_subset = resample(class_subset,
                                            replace=True,  # Sample with replacement
                                            n_samples=majority_count,  # Match majority class count
                                            random_state=9)  # For reproducibility
            resampled_data.append(resampled_class_subset)

    print("Downsampling majority.")
    # Downsample the majority class
    majority_class_subset = patch_df[patch_df['patch_Label'] == majority_class]
    downsampled_majority_class_subset = resample(majority_class_subset,
                                                replace=False,  # Sample without replacement
                                                n_samples=majority_count,  # Match the majority class count
                                                random_state=9)  # For reproducibility
    resampled_data.append(downsampled_majority_class_subset)

    # Combine the resampled DataFrames
    balanced_data = pd.concat(resampled_data).reset_index(drop=True)
    # Shuffle the balanced dataset
    data = balanced_data.sample(frac=1, random_state=9).reset_index(drop=True)
    # data = balanced_data[~balanced_data.index.isin(data_initial.index)]
    # Check the new class distribution
    print("Balanced class distribution:\n", balanced_data['patch_Label'].value_counts())
    del patch_df
    del balanced_data
    del resampled_data
    # del data_initial
    
    
    # ********************************************** BUILD DATASET ***************************************************
    train,valid = train_test_split(data,test_size=0.4,random_state=42)
    train = train.reset_index(drop=True)
    valid = valid.reset_index(drop=True)
    
    del data
    
    # paths = glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\*.mat")
    # hsi_paths = []
    # for path in paths:
    #     name = path.split("\\")[-1]
    #     if "gt" not in name:
    #         hsi_paths.append(path)
    # hsi_dict = {}
    # for path in hsi_paths:
    #     hsi_dict[path] = np.transpose(loadmat(path)['HDR'][:],axes=(1,2,0))
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
    
    paths = glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\*.mat")
    hsi_paths = []
    for path in paths:
        name = path.split("\\")[-1]
        if "gt" not in name:
            hsi_paths.append(path)
    hsi_dict = {}
    for path in hsi_paths:
        hsi_dict[path] = np.transpose(loadmat(path)['HDR'][:],axes=(1,2,0)).shape
    mmaps = {}
    for path,shape in hsi_dict.items():
        mmaps[path] = np.memmap(path, dtype='float32', mode='r', shape=shape)
        
    # pickle.dump(mmaps,open("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Vision_Transformer\\HSI_Transformer\\mmaps_dict.pkl",'wb'))
        
        
    print("Preparing the Dataloaders\n")
    window_size = 15
    train_data = Minerals_Dataset(train,mmaps,window_size=window_size,transforms=None)  
    test_data = Minerals_Dataset(valid,mmaps,window_size=window_size,transforms=None)
    train_dl = DataLoader(train_data,batch_size=512,num_workers=2,shuffle=True,pin_memory=True,drop_last=False)
    valid_dl = DataLoader(test_data,batch_size=512,num_workers=2,shuffle=False,pin_memory=True,drop_last=False)
    
    for img,label in valid_dl:
        print("Printing dataloader shapes")
        print(img.shape)
        print(label.shape)
        print(img.dtype)
        print((img.element_size()*img.numel())/(1024*1024),"MB")
        break
    
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device active: ",device)
    gc_collect()

    # # ************************* MODEL INITIALIZATION *****************************
    model = MViT(window_size,76,3,512,3,8,0.1,256,256,"gelu")
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Total number of parameters: {total_params}")
    # torch.save(model.state_dict(),"D:\\HSI Project\\Updated_Work\\HSI_Classification\\Vision_Transformer\\HSI_Transformer\\MViT_dummy.pth")
    model = model.to(device)
    
    # ************************************************** Training *************************************************
    epoch = 10
    model,train_loss,val_loss,train_fscore,val_fscore = train_model(model,train_dl,valid_dl,epoch,device,1e-4,1e-6)
    
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
    # ax[1].plot(epochs, train_fscore, label='Train F-Score',color='blue')
    ax[1].plot(epochs, val_fscore, label='Validation F-Score',color='orange')

    # Add labels and title
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('F-Score')
    ax[1].set_title('Train and Validation F-Scores')

    # Add legend
    # ax[1].legend()

    # Show plot
    plt.show()
        
    return model

if __name__ == '__main__':
    # mp.set_start_method('spawn')
    model = train()
    
    