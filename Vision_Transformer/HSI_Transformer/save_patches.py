import numpy as np
from scipy.io import loadmat,savemat
from utils import padWithZeros,createImageCubes, generate_binary_mask
import pandas as pd
from tqdm import tqdm
from glob import glob
import h5py

def save_patches():
    # Patch Saving 
    df = pd.read_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_Mapped.csv")
    minerals = df['Mineral'].unique().tolist()
    minerals_classes = {}
    for i,min in enumerate(minerals):
        minerals_classes[min] = i+1

    # patch_df = pd.DataFrame(columns=["ID","EXP","patch_id","hsi_path","patch_Label"])
    # data_list = []
    # for _, row in tqdm(df.iterrows(),total=len(df)):
    #     hsi_path = row['HSI_Path']
    #     mask_path = row['MASK_Path']
    #     exp = row['Experiment']
    #     ID = row['ID']
    #     mineral = row['Mineral']
    #     hsi = np.transpose(loadmat(hsi_path)['HDR'][:],axes=(1,2,0))
    #     mask = loadmat(mask_path)['MASK'][:]
    #     mask = np.where(mask==255,minerals_classes[mineral],0) 
    #     for id in range(hsi.shape[0]*hsi.shape[1]):
    #         x,y = id//hsi.shape[1],id%hsi.shape[1]
    #         label = mask[x,y]
    #         data = {
    #             'ID':ID,
    #             'EXP':exp,
    #             'patch_id':id,
    #             'hsi_path':hsi_path,
    #             'patch_Label':label
    #         }
    #         data_list.append(data)
    # patch_df = pd.DataFrame(data_list)
    # patch_df.to_csv("E:\\Patches_Data\\Patch_df_for_any_window_size.csv",index=False)
    
    patch_df = pd.read_csv("E:\\Patches_Data\\Patch_df_for_any_window_size.csv")
    print(len(sorted(patch_df['ID'].unique())))
    # print(patch_df[patch_df['ID']==5])
    
    # hsi_path = df[(df["ID"]==5) & (df['Experiment']=='B')]['HSI_Path'].iloc[0]
    # print(hsi_path)
    # print(np.transpose(loadmat(hsi_path)['HDR'][:],axes=(1,2,0)).shape)



    # print(df.loc[0,'HSI_Path'])

    #**************************** SAVING PATCHES(WITH BACKGROUND) ******************************

    # patch_df = pd.DataFrame(columns=["ID","EXP","patch_id","patch_path","patch_Label"])
    # for _, row in tqdm(df.iterrows(),total=len(df)):
    #     hsi_path = row['HSI_Path']
    #     mask_path = row['MASK_Path']
    #     exp = row['Experiment']
    #     ID = row['ID']
    #     mineral = row['Mineral']
    #     hsi = np.transpose(loadmat(hsi_path)['HDR'][:],axes=(1,2,0))
    #     print(hsi_path)
    #     # np.save("E:\\demo_patches_data\\HSI_Data.npy",hsi)
    #     mask = loadmat(mask_path)['MASK'][:]
    #     mask = np.where(mask==255,minerals_classes[mineral],0) 
    #     print(hsi.shape)
        # patches,patchesLabels = createImageCubes(hsi,mask,5,False) # Taking 5x5 Patches. 
        # np.save("E:\\demo_patches_data\\pc2\\{ID}_{exp}.npy",patches)
        # np.save("E:\\demo_patches_data\\pc2\\{ID}_{exp}_labels.npy",patchesLabels)

    # loaded_array_npz = np.load("E:\\demo_patches_data\\pc2\\all_data.npz")
    # print(len(loaded_array_npz.keys()))
    # print(loaded_array_npz)
    



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

