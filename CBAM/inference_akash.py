import torch
# from Hybrid_CNN_Model import HybridCNN
from Cbam2 import CBAM_Model
import numpy as np
from utils import fill_nan
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import h5py 
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

class Minerals_Inference_Dataset(Dataset):
    def __init__(self, patches):
        self.patches = patches

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        return torch.tensor(self.patches[idx], dtype=torch.float32)

def create_patches(hdr_data_padded, patch_size, stride):
    patches = []
    positions = []
    padding = patch_size // 2
    final_padding = 1
    for x in tqdm(range(padding, hdr_data_padded.shape[1] - padding, stride)):
        for y in range(padding, hdr_data_padded.shape[2] - padding, stride):
            patch = np.transpose(hdr_data_padded[:, x - padding:x + padding + 1, y - padding:y + padding + 1], axes=(1, 2, 0))
            patch_padded = np.pad(patch, ((final_padding, final_padding), (final_padding, final_padding), (0, 0)), mode='constant')
            patches.append(patch_padded)
            # print(patch_padded.shape)
            positions.append((x - padding, y - padding))
    return patches, positions

def reconstruct_image(labels, positions, image_shape):
    reconstructed_image = np.zeros(image_shape)
    for label, (x, y) in zip(labels, positions):
        reconstructed_image[x, y] = label
    return reconstructed_image

def inference(hdr_data, patch_size, model, stride=1, device='cpu'):
    padding = patch_size // 2
    hdr_data_padded = np.pad(hdr_data, ((0, 0), (padding, padding), (padding, padding)), mode='constant')
    
    patches, positions = create_patches(hdr_data_padded, patch_size, stride)
    dataset = Minerals_Inference_Dataset(patches)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False, pin_memory=True, drop_last=False)
    
    model.eval()
    all_labels = []
    
    with torch.no_grad():
        for patch_batch in tqdm(dataloader):
            # print(patch_batch.shape)
            patch_batch = patch_batch.to(device).permute(0,3,1,2)
            # print(patch_batch.shape)
            label_pred = model(patch_batch)
            # print(label_pred.shape)
            labels = torch.argmax(label_pred, dim=1).cpu().detach().numpy()
            all_labels.extend(labels)
    
    output_image = reconstruct_image(all_labels, positions, hdr_data.shape[1:])
    return output_image

if __name__=="__main__":    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device active: ",device)

    hdr_path = "D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\HDR_Data\\Extracted_Folder\\0002\\A.hdr.h5"
    hdr_data = h5py.File(hdr_path,'r')['hdr'][:]

    model_path = "D:\\HSI Project\\Updated_Work\\HSI_Classification\\CBAM\\cbam_model.pth"
    # model = HybridCNN(256,7,128,32,76)
    # model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model = CBAM_Model(256,128,64,32,76)
    # model.load_state_dict(torch.load(model_path))
    model.load_state_dict(torch.load(model_path))

    patch_size = 5

    mask = inference(hdr_data,patch_size,model)

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
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    im1 = axs[0].imshow(hdr_data[0], cmap='gray', aspect='auto')
    axs[0].set_title('Original HDR Data Slice')
    fig.colorbar(im1, ax=axs[0])

    cmap = ListedColormap(colors)
    im2 = axs[1].imshow(mask, cmap=cmap, aspect='auto')
    axs[1].set_title('Reconstructed Mask')
    fig.colorbar(im2, ax=axs[1])

    plt.show()