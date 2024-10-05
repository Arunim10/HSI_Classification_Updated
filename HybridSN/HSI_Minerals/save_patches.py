import numpy as np
from scipy.io import loadmat,savemat
from utils import padWithZeros,createImageCubes, generate_binary_mask
import pandas as pd
from tqdm import tqdm
from glob import glob


def save_patches():
    # Patch Saving 
    df = pd.read_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_Mapped.csv")
    minerals = df['Mineral'].unique().tolist()
    minerals_classes = {}
    for i,min in enumerate(minerals):
        minerals_classes[min] = i+1

    #**************************** SAVING PATCHES(WITH BACKGROUND) ******************************

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

    #*********************************** SAVING PATHS OF PATCHES(WITH BACKGROUND) TO A DATAFRAME *******************************************

    # patch_lists = glob("E:\\Patches_Data\\Labels\\*")
    # data_list = []
    # for path in tqdm(patch_lists,total=len(patch_lists)):
    #     name = path.split('\\')[-1]
    #     patch_path = f"E:\\Patches_Data\\Patches\\{name}"
    #     label = np.load(path)
    #     data_list.append({'Patch_Path':patch_path,'Label':label})
    # patch_df = pd.DataFrame(data_list)
    # patch_df.to_csv("E:\\Patches_Data\\Patch_5x5_df.csv",index=False)
    
    # # print("Patches Saved Successfully !!")
    # print("Patch Dataframe saved successfully !!")
    
if __name__=="__main__":
    save_patches()

