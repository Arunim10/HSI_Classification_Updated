import numpy as np
from scipy.io import loadmat,savemat
from utils import padWithZeros,createImageCubes, generate_binary_mask
import pandas as pd
from tqdm import tqdm
from glob import glob

# Patch Saving 
df = pd.read_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_Mapped.csv")
df = df.drop(df[df['Mineral']=='Background'].index).reset_index(drop=True)
print(df['Mineral'].unique())
minerals = df['Mineral'].unique().tolist()
minerals_classes = {}
for i,min in enumerate(minerals):
    minerals_classes[min] = i+1
    

# print(df['ID'].nunique())
# print(df.info())



#**************************** SAVING PATCHES WIHOUT TAKING BACKGROUND INTO ACCOUNT ******************************

# patch_df = pd.DataFrame(columns=["ID","EXP","patch_id","patch_path","patch_Label"])
# for _, row in tqdm(df.iterrows(),total=len(df)):
#     hsi_path = row['HSI_Path']
#     mask_path = row['MASK_Path']
#     exp = row['Experiment']
#     ID = row['ID']
#     mineral = row['Mineral']
#     hsi = np.transpose(loadmat(hsi_path)['HDR'][:],axes=(1,2,0))
#     mask = loadmat(mask_path)['MASK'][:]
#     mask = np.where(mask==255,minerals_classes[mineral],0) 
#     patches,patchesLabels = createImageCubes(hsi,mask,5,False) # Taking 5x5 Patches. 
#     for id in range(patches.shape[0]):
#         save_patch_path = f"E:\\Patches_Data\\Patches\\{ID}_{exp}_{id}.npy"
#         save_label_path = f"E:\\Patches_Data\\Labels\\{ID}_{exp}_{id}.npy"
#         np.save(arr=patches[id].astype(np.float16),file=save_patch_path)
#         np.save(arr=patchesLabels[id].astype(np.uint8),file=save_label_path)

#*********************************** SAVING PATHS OF PATCHES(WITHOUT BACKGROUND) TO A DATAFRAME *******************************************

# print(len(glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Patches_Data\\*")))
patch_lists = glob("E:\\Patches_Data\\Labels\\*")
data_list = []
for path in tqdm(patch_lists,total=len(patch_lists)):
    name = path.split('\\')[-1]
    patch_path = f"E:\\Patches_Data\\Patches\\{name}"
    label = np.load(path)
    data_list.append({'Patch_Path':patch_path,'Label':label})
patch_df = pd.DataFrame(data_list)
patch_df.to_csv("E:\\Patches_Data\\Patch_5x5_df.csv",index=False)

    
# ***************************** SAVING PATCHES FOR BACKGROUND ********************************************

# hsi_path_bg = df[df['Mineral']=="Background"].loc[152]['HSI_Path']
# mask_path_bg = df[df['Mineral']=="Background"].loc[152]['MASK_Path']
# id_bg = df[df['Mineral']=="Background"].loc[152]['ID']
# exp_bg = df[df['Mineral']=="Background"].loc[152]['Experiment']
# mineral = 'Background'
# hsi = np.transpose(loadmat(hsi_path_bg)['HDR'][:],axes=(1,2,0))
# mask = loadmat(mask_path_bg)['MASK'][:]
# mask = np.where(mask==255,minerals_classes[mineral],0) 
# patches,patchesLabels = createImageCubes(hsi,mask,3,True) ## patchLabels will have '0' label for a mineral not background, for background handle later when takn background into account
# for id in range(patches.shape[0]):
#     save_patch_path = f'D:\\HSI Project\\Updated_Work\\HSI_Classification\\RPNet_RF\\Minerals\\Patches_3\\Patches\\{id_bg}_{exp_bg}_{id}.npy'
#     save_label_path = f'D:\\HSI Project\\Updated_Work\\HSI_Classification\\RPNet_RF\\Minerals\\Patches_3\\Labels\\{id_bg}_{exp_bg}_{id}.npy'
#     np.save(arr=patches[id].astype(np.float16),file=save_patch_path)
#     np.save(arr=patchesLabels[id].astype(np.uint8),file=save_label_path)

#********************************** APPENDING PATHS OF BACKGROUND TO ABOVE CREATED DATAFRAME *********************************************

# patch_df = pd.read_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\RPNet_RF\\Minerals\\patch_3_df.csv")
# patch_lists = glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\RPNet_RF\\Minerals\\Patches_3\\Labels\\*")
# data_list_bg = []
# for path in tqdm(patch_lists,total=len(patch_lists)):
#     if '130_A' in path:
#         name = path.split('\\')[-1]
#         patch_path = f"D:\\HSI Project\\Updated_Work\\HSI_Classification\\RPNet_RF\\Minerals\\Patches_3\\Patches\\{name}"
#         label = np.load(path)
#         data_list_bg.append({'Patch_Path':patch_path,'Label':label})
# # patch_df = pd.DataFrame(data_list)
# bg_patch_df = pd.DataFrame(data_list_bg)
# patch_df = pd.concat([patch_df,bg_patch_df],ignore_index=True)
# # print(patch_df.tail())
# patch_df.to_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\RPNet_RF\\Minerals\\patch_3_df.csv",index=False)

