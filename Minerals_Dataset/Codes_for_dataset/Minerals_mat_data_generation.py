from glob import glob
import numpy as np
import h5py
import cv2
import scipy
from scipy.io import loadmat
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_binary_mask(hsi_data,mask_data):
    hsi_data = np.nan_to_num(hsi_data, nan=0)
    hsi_temp = hsi_data[0].copy() ## Selecting first channel
    mask_data = np.nan_to_num(mask_data, nan=0)
    mask = mask_data[0].copy() ## Selecting first channel
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            val = mask[i,j]
            if (val>0):
                mask_temp = (hsi_temp==val)
                hsi_temp[mask_temp] = 255
    hsi_temp = np.where((hsi_temp>0)&(hsi_temp<255),0,hsi_temp)
    
    hsi_temp = hsi_temp.astype(np.uint8)
    num_labels, labeled_image = cv2.connectedComponents(hsi_temp)
    sizes = np.bincount(labeled_image.flatten())
    threshold_size = 20
    new_mask = np.zeros_like(labeled_image)
    for label, size in enumerate(sizes):
        if size >= threshold_size and label != 0:  # Exclude background label (0)
            new_mask[labeled_image == label] = 255            
    return new_mask

def spatial_avg(data,coordinates,kernel_size=3):
    (x,y,c) = coordinates
    # data = patch[:,:,c].copy()
    while (np.isnan(data[x,y]) and kernel_size<=5):
        max_x = data.shape[0]-1
        max_y = data.shape[1]-1
        start_x = max(0,x-kernel_size//2)
        start_y = max(0,y-kernel_size//2)
        end_x = min(max_x,x+kernel_size//2)+1
        end_y = min(max_y,y+kernel_size//2)+1
        neighbours = []
        for i in range(start_x,end_x):
            for j in range(start_y,end_y):
                neighbours.append(data[i,j])
        # Check if all elements are nan
        all_nan = all(np.isnan(x) for x in neighbours)
        if all_nan:
            kernel_size+=2
            continue
        data[x,y] = np.nanmean(neighbours)
    return data

def fill_nan_hdr(hsi_img):
    # hsi_img: shape (h,w,c)
    hsi = hsi_img.copy()
    ind = np.transpose(np.where(np.isnan(hsi)))
    ind = np.array(sorted(ind,key=lambda x: (x[2],x[0],x[1])))
    for (x,y,c) in ind:
        if (c==0):
            if not np.isnan(hsi[x,y,c+1]):
                hsi[x,y,c] = hsi[x,y,c+1]
            else:
                hsi[:,:,c] = spatial_avg(hsi[:,:,c],(x,y,c))
        elif (c==255):
            if not np.isnan(hsi[x,y,c-1]):
                hsi[x,y,c] = hsi[x,y,c-1]
            else:
                hsi[:,:,c] = spatial_avg(hsi[:,:,c],(x,y,c))
        else:
            if (not np.isnan(hsi[x,y,c+1])) and (not np.isnan(hsi[x,y,c-1])):
                hsi[x,y,c] = np.mean([hsi[x,y,c+1],hsi[x,y,c-1]])
            elif not np.isnan(hsi[x,y,c-1]):
                hsi[x,y,c] = hsi[x,y,c-1]
            elif not np.isnan(hsi[x,y,c+1]):
                hsi[x,y,c] = hsi[x,y,c+1]
            else:
                hsi[:,:,c] = spatial_avg(hsi[:,:,c],(x,y,c))
    return hsi

def main():
    # hdr_files = list(glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\HDR_Data\\Extracted_Folder\\*\\*.hdr.h5"))
    # masked_hdr_files = list(glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Masked_HRD_Data\\Extracted_Folder\\*\\*.mhdr.h5"))
    # for (hdr_pth,mask_pth) in tqdm(zip(hdr_files,masked_hdr_files),total=len(hdr_files)):
    #     mask = h5py.File(mask_pth,'r')['hdr'][:]
    #     hsi = h5py.File(hdr_pth,'r')['hdr'][:] # (256,320,410)
        
    #     ## Handling Nan Values here 
    #     hsi = np.transpose(hsi,axes=(1,2,0))
    #     hsi = fill_nan_hdr(hsi)
    #     hsi = np.transpose(hsi,axes=(2,0,1))  # (256,320,410)
        
    #     mask = np.nan_to_num(mask, nan=0)
    #     try:
    #         new_mask = generate_binary_mask(hsi,mask)
    #         hsi_data = {'HDR': hsi}
    #         mask_data = {'MASK':new_mask}
    #         hsi_name = f'{hdr_pth.split("\\")[-2]}_{hdr_pth.split("\\")[-1][0]}.mat'
    #         mask_name = f'{hdr_pth.split("\\")[-2]}_{hdr_pth.split("\\")[-1][0]}_gt.mat'
    #         hsi_mat_path = f'D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\{hsi_name}'
    #         mask_mat_path = f'D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\{mask_name}'
    #         scipy.io.savemat(hsi_mat_path, hsi_data)
    #         scipy.io.savemat(mask_mat_path, mask_data)
            
    #     except FileNotFoundError:
    #         continue
        
        nan_img_even_after_fill_nan_function = {
            "89_A" : [0,0],
            "89_B" : [0,0],
            "90_A" : [0,0],
            "87_A" : [[0,0],[0,1]],
            "87_B" : [0,0],
            "86_A" : [[0,0],[0,1]],
            "82_A" : [0,0],
            "91_A" : [0,0],
            "91_B" : [0,0],
            "92_A" : [0,0],
            "92_B" : [0,0],
            "83_A" : [[0,0],[0,1]],
            "84_A" : [0,0],
            "85_A" : [0,0],
            "93_A" : [[0,0],[0,1]]
        }
        for name,indices in nan_img_even_after_fill_nan_function.items():
            path = f"D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\00{name}.mat"
            hsi = loadmat(path)['HDR']
            if len(indices)==1:
                hsi[:,0,0] = np.mean([hsi[:,0,1],hsi[:,1,0],hsi[:,1,1]],axis=0)
            else:
                hsi[:,0,1] = np.mean([hsi[:,0,2],hsi[:,1,1],hsi[:,1,2]],axis=0)
                hsi[:,0,0] = np.mean([hsi[:,0,1],hsi[:,1,0],hsi[:,1,1]],axis=0)
            ind = np.where(np.isnan(hsi))
            print(ind)
            hsi_data = {'HDR': hsi}
            scipy.io.savemat(path, hsi_data)

    # print("Mat Files for Minerals are generated and saved successfully !!")
    
if __name__ == "__main__":
    main()    
