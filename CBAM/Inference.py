import torch
from CBAM_Model import CBAM_Model
import numpy as np
from utils import fill_nan
import torch.nn.functional as F
from torch.utils.data import DataLoader
import h5py 
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
import cv2
import pandas as pd
from scipy.io import loadmat

df = pd.read_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_Mapped.csv")
# df = df.drop(df[df['Mineral']=='Background'].index).reset_index(drop=True)
minerals = df['Mineral'].unique().tolist()
minerals_classes = {'Background':0}
for i,mine in enumerate(minerals):
    minerals_classes[mine] = i+1
print(minerals_classes)

class Minerals_Inference_Dataset(torch.utils.data.Dataset):
    def __init__(self,patches):
        super().__init__()
        self.patches = patches
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self,idx):
        
        patch = self.patches[idx]
        # Handling Nan Values
        # ind = np.where(np.isnan(patch))
        # for (x,y,c) in zip(ind[0],ind[1],ind[2]):
        #     patch = fill_nan(patch,(x,y,c)) 
        patch = np.pad(patch,((1,1),(1,1),(0,0)),mode='constant')
        return torch.tensor(patch).permute(2,0,1)
    
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device active: ",device)

def spatial_avg(data,coordinates,kernel_size=3):
    (x,y,c) = coordinates
    # data = patch[:,:,c].copy()
    # print(x," ",y," ",c)
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

def create_patches(hdr_data_padded,patch_size,row,col,padding):
    # Selecting patch_size x patch_size shape patches from hdr_data
    start_x = patch_size//2
    start_y = patch_size//2
    end_x = row + (patch_size//2)
    end_y = col + (patch_size//2)
    
    patches = []
    positions = []
    for x in range(start_x,end_x):
        for y in range(start_y,end_y):
            patch = np.transpose(hdr_data_padded[:,x-padding:x+padding+1,y-padding:y+padding+1],axes=(1,2,0))
            patches.append(patch)
            positions.append((x-padding,y-padding))
    return patches,positions

def reconstruct_image(labels, positions, image_shape):
    reconstructed_image = np.zeros(image_shape)
    for label, (x, y) in zip(labels, positions):
        reconstructed_image[x, y] = label
    return reconstructed_image

def inference(hdr_data: np.ndarray, patch_size: int,model: CBAM_Model): 
    # hdr_data will be of shape (c,h,w) 
    channel,row,col = hdr_data.shape[0],hdr_data.shape[1],hdr_data.shape[2]
    padding = patch_size//2
    hdr_data_padded = np.pad(hdr_data,((0,0),(padding,padding),(padding,padding)),mode='constant')
    mask = np.zeros((row,col))
    patches,positions = create_patches(hdr_data_padded,patch_size,row,col,padding)
    dataset = Minerals_Inference_Dataset(patches)
    dataloader = DataLoader(dataset,batch_size=128,shuffle=False,pin_memory=True,drop_last=False)
    
    model = model.to(device)
    model.eval()
    all_labels = []
    with torch.no_grad():
        for patch in tqdm(dataloader):
            patch  = patch.to(device)
            label_pred = model(patch.to(torch.float32))
            labels = torch.argmax(label_pred,dim=1)
            # condition = (label_pred <= 0.5).all(dim=1)
            # labels[condition] = 0
            all_labels.extend(labels.cpu().detach().numpy())
    
    output_image = reconstruct_image(all_labels, positions, (hdr_data.shape[1],hdr_data.shape[2]))
    
    return output_image    

hdr_path = "D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\HDR_Data\\Extracted_Folder\\0128\\A.hdr.h5"
hdr_data = h5py.File(hdr_path,'r')['hdr'][:]
hsi = np.transpose(hdr_data,axes=(1,2,0))
hsi = fill_nan_hdr(hsi)
hsi = np.transpose(hsi,axes=(2,0,1))  # (256,320,410)

mask_act_path = "D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\0128_A_gt.mat"
mask_actual = loadmat(mask_act_path)['MASK']
model_path = "D:\\HSI Project\\Updated_Work\\HSI_Classification\\CBAM\\cbam_model3.pth"
model = CBAM_Model(256,128,64,32,76)
model.load_state_dict(torch.load(model_path))

patch_size = 5

mask = inference(hsi,patch_size,model)

colors = [
    "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF", "#00FFFF",
    "#800000", "#008000", "#000080", "#808000", "#800080", "#008080",
    "#C00000", "#00C000", "#0000C0", "#C0C000", "#C000C0", "#00C0C0",
    "#400000", "#004000", "#000040", "#404000", "#400040", "#004040",
    "#FF4000", "#40FF00", "#0040FF", "#FF0040", "#40FF40", "#FF40FF",
    "#FF8040", "#80FF40", "#4080FF", "#FF4080", "#40FF80", "#FF80FF",
    "#800040", "#008040", "#400080", "#804000", "#408000", "#004080",
    "#804080", "#808040", "#408080", "#808080", "#C04040", "#40C040",
    "#4040C0", "#C040C0", "#40C0C0", "#C0C040", "#C08040", "#80C040",
    "#4080C0", "#C04080", "#40C080", "#C08080", "#C0C080", "#80C080",
    "#C0C0C0", "#400000", "#004000", "#000040", "#404000", "#400040",
    "#004040", "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    "#00FFFF", "#800000", "#008000", "#000080"
]

# np.save(arr=mask,file="D:\\HSI Project\\Updated_Work\\HSI_Classification\\CBAM\\mask.npy")
# mask = np.load("D:\\HSI Project\\Updated_Work\\HSI_Classification\\CBAM\\mask.npy")
cmap = ListedColormap(colors)
fig,ax = plt.subplots(1,3,figsize=(15,5))
cax1 = ax[0].imshow(mask,cmap=cmap,aspect='auto')
fig.colorbar(cax1, ax=ax[0])
ax[1].imshow(hdr_data[0,:,:],aspect='auto')
ax[2].imshow(mask_actual,aspect='auto')
plt.show()    
