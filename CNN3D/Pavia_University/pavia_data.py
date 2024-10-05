## IMPORTING LIBRARIES
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import random
from tqdm import tqdm

## LOADING DATA
# paviaU = loadmat(r'D:\HSI Project\Updated_Work\Paper_Model_Building_Pytorch_Notebooks\CNN3D\PaviaU.mat')['paviaU']
# paviaU_gt = loadmat(r'D:\HSI Project\Updated_Work\Paper_Model_Building_Pytorch_Notebooks\CNN3D\PaviaU_gt.mat')['paviaU_gt']
# print(paviaU.shape,paviaU_gt.shape)

## PLOTTING DATA
# fig,ax = plt.subplots(1,2,figsize=(10,10))
# ax[0].imshow(paviaU[:,:,20],cmap='gray') # 20 is just a random number
# ax[0].set_title('HSI Data')
# ax[1].imshow(paviaU_gt)
# ax[1].set_title('Ground Truth')
# plt.show()


## Building Dataset - n x n neighbourhood kernel
## We are taking n=5 here as given by the paper which performs best

# def convert(dicti,task):
#     X = []
#     Y = []
#     for labels in dicti:
#         for ten in dicti[labels]:
#             X.append(ten)
#             if (task=="train"):
#                 Y.append(labels[:-6])
#             else:
#                 Y.append(labels[:-4])
#     return X,Y

# def train_test_split_data(kernel_size,data,data_gt):
#     k = kernel_size
#     pU = torch.tensor(np.array(data.copy(),dtype=np.float64))
#     pU = pU.unsqueeze(0).permute(0,3,1,2)
#     padding = [k//2,k//2,k//2,k//2]
#     padd = F.pad(pU,padding).numpy()
#     paviU_data = {}
#     for i in range(1,10):
#         paviU_data[f"label_{i}"] = []
#     for i in range(pU.shape[2]):
#         for j in range(pU.shape[3]):
#             temp = padd[:,:,i:i+k//2+k//2+1,j:j+k//2+k//2+1]
#             label = paviaU_gt[i,j]
#             if label!=0:
#                 paviU_data[f"label_{label}"].append(temp)
                
        
#     paviaU_train_test_split = {}
#     for labels in paviU_data:
#         print(labels)
#         train = random.sample(paviU_data[labels],300)
#         val = []
#         for ten1 in tqdm(paviU_data[labels],total=len(paviU_data[labels])):
#             temp = False
#             for ten2 in train:
#                 if ((ten1==ten2).all().item()):
#                     temp=True
#                     break
#             if temp==False:
#                 val.append(ten1)
#         paviaU_train_test_split[f'{labels}_train'] = train
#         paviaU_train_test_split[f'{labels}_val'] = val  
        
#     train_dict = {}
#     test_dict = {}
#     for labels in paviaU_train_test_split:
#         if (labels[-5:]=="train"):
#             train_dict[labels] = paviaU_train_test_split[labels]
#         else:
#             test_dict[labels] = paviaU_train_test_split[labels]
            
            
#     train_X,train_Y = convert(train_dict,"train")
#     test_X,test_Y = convert(test_dict,"test")
    
#     return (train_X,train_Y),(test_X,test_Y)

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

def splitTrainTestSet(X, y, testRatio, randomState=345):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testRatio, random_state=randomState,
                                                        stratify=y)
    return X_train, X_test, y_train, y_test

def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0],X.shape[1], numComponents))
    return newX, pca

def padWithZeros(X, margin=2):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def createImageCubes(X, y, windowSize=5, removeZeroLabels = True):
    margin = int((windowSize - 1) / 2)
    zeroPaddedX = padWithZeros(X, margin=margin)
    # split patches
    patchesData = np.zeros((X.shape[0] * X.shape[1], windowSize, windowSize, X.shape[2]))
    patchesLabels = np.zeros((X.shape[0] * X.shape[1]))
    patchIndex = 0
    for r in range(margin, zeroPaddedX.shape[0] - margin):
        for c in range(margin, zeroPaddedX.shape[1] - margin):
            patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]   
            patchesData[patchIndex, :, :, :] = patch
            patchesLabels[patchIndex] = y[r-margin, c-margin]
            patchIndex = patchIndex + 1
    if removeZeroLabels:
        patchesData = patchesData[patchesLabels>0,:,:,:]
        patchesLabels = patchesLabels[patchesLabels>0]
        patchesLabels -= 1
    return patchesData, patchesLabels

def train_test_split_data(kernel_size,data,data_gt,num_components):
    data_PCA,pca = applyPCA(data,num_components)
    data_patches, data_patchLabels = createImageCubes(data_PCA, data_gt, windowSize=kernel_size)
    Xtrain, Xtest, ytrain, ytest = splitTrainTestSet(data_patches, data_patchLabels, 0.95)
    # ytrain_onehot = torch.nn.functional.one_hot(torch.tensor(ytrain).long(), num_classes=9)
    # ytest_onehot = torch.nn.functional.one_hot(torch.tensor(ytest).long(), num_classes=9)
    return (Xtrain,ytrain),(Xtest,ytest)