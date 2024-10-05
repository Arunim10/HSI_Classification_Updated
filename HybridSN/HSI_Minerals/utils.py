import numpy as np
import torch
import cv2
from scipy.io import loadmat

# ************************************* EXTRACTING PATCHES OF ANY SIZE **********************************
def patch_from_hsi(id,hsi_path,window_size):
    
    hsi = np.transpose(loadmat(hsi_path)['HDR'][:],axes=(1,2,0))
    
    row,col,channels = hsi.shape
    x_hsi,y_hsi = id//col,id%col
    margin = int((window_size-1)/2)
    
    patch = np.zeros((window_size,window_size,channels))
    
    lower_bound = [max(0,x_hsi-margin),max(0,y_hsi-margin)]
    upper_bound = [min(row-1,x_hsi+margin),min(col-1,y_hsi+margin)]
    
    diff_lower = [abs(x_hsi-lower_bound[0]),abs(y_hsi-lower_bound[1])]
    diff_upper = [abs(upper_bound[0]-x_hsi),abs(upper_bound[1]-y_hsi)]
    
    patch_lower_bound = [margin-diff_lower[0],margin-diff_lower[1]]
    patch_upper_bound = [margin+diff_upper[0],margin+diff_upper[1]]
    
    patch[patch_lower_bound[0]:patch_upper_bound[0]+1,patch_lower_bound[1]:patch_upper_bound[1]+1,:] = hsi[lower_bound[0]:upper_bound[0]+1,lower_bound[1]:upper_bound[1]+1,:]
    
    return patch

# *************** FILLING NAN VALUES *********************

def spatial_avg(patch,coordinates,kernel_size=3):
    (x,y,c) = coordinates
    data = patch[:,:,c].copy()
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
        patch[:,:,c] = data
    return patch
                
def fill_nan(patch,coordinates): # coordinats = (x,y,c)
    data = patch.copy()
    (x,y,c) = coordinates
    if (c==0):
        if not np.isnan(data[x,y,c+1]):
            data[x,y,c] = data[x,y,c+1]
        else:
            data = spatial_avg(data,(x,y,c))
    elif (c==255):
        if not np.isnan(data[x,y,c-1]):
            data[x,y,c] = data[x,y,c-1]
        else:
            data = spatial_avg(data,(x,y,c))
    else:
        if (not np.isnan(data[x,y,c+1])) and (not np.isnan(data[x,y,c-1])):
            data[x,y,c] = np.mean([data[x,y,c+1],data[x,y,c-1]])
        elif not np.isnan(data[x,y,c-1]):
            data[x,y,c] = data[x,y,c-1]
        elif not np.isnan(data[x,y,c+1]):
            data[x,y,c] = data[x,y,c+1]
        else:
            data = spatial_avg(data,(x,y,c))
    return data

# ******************************** CREATING BINARY MASKS ***************************

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

# ********************************** VOXELS FORMATION ***************************************

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

# ******************************** EVALUATION METRICS ********************************
from sklearn.metrics import f1_score
def f_score(y_true,y_pred):
    ## labels = [0,5,2,1,4]
    ## predictions = [0,5,1,4,2]
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    scores = f1_score(y_true, y_pred, average=None)
    return scores
def accuracy(y_true,y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    acc = (y_true==y_pred).sum()/len(y_true)
    return acc

# ******************************** EVALUATION FUNCTION *********************************
from tqdm import tqdm
def evaluate_model(model,dl,device,name):
    torch.manual_seed(42)
    model = model.to(device)
    pred = []
    with torch.no_grad():
        model.eval()
        losses = []
        targets = []
        with tqdm(dl,desc=f'{name}',mininterval=30) as progress:
            for i,(spec_values,labels) in enumerate(progress):
                    spec_values = spec_values.to(device)
                    labels = labels.to(device,dtype=torch.int64)
                    label_pred = model(spec_values.to(torch.float32))
                    label_pred = label_pred.squeeze()
                    labels = labels.squeeze()
                    loss = torch.nn.functional.cross_entropy(label_pred,labels).item()
                    losses.append(loss)
                    lab_out = torch.argmax(label_pred,dim=1)
                    pred.append(lab_out.cpu().numpy())
                    targets.append(labels.cpu().numpy())
                    f1_temp = f_score(labels.cpu().numpy(),lab_out.cpu().numpy())
                    acc_temp = accuracy(labels.cpu().numpy(),lab_out.cpu().numpy())
                    progress.set_description(f'{name}: loss:{loss:.4f},f_sc:{np.mean(f1_temp):.4f},acc:{acc_temp:.4f}')
    pred = np.concatenate(pred)
    targets = np.concatenate(targets)
    f1_sc = f_score(targets,pred)
    acc = accuracy(targets,pred)
    return pred,targets,losses,f1_sc,acc

# ***************************************** TRAINING FUNCTION *****************************************
import gc
def gc_collect():
    gc.collect()
    torch.cuda.empty_cache()

import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
def train_model(model,train_dl,test_dl,epochs,device,lr,wd):
    torch.manual_seed(42)
    optim = torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_dl))
    best_eval_score = 0
    val_loss = []
    train_loss = []
    val_fscore = []
    train_fscore = []
    for epoch in range(epochs):
        print("Epoch ",epoch)
        model.train()
        with tqdm(train_dl,desc='Train',mininterval=30) as train_progress:
            for i,(spec_values,labels) in enumerate(train_progress):
                    optim.zero_grad()
                    spec_values = spec_values.to(device)
                    labels = labels.to(device,dtype=torch.int64)
                    label_pred = model(spec_values.to(torch.float32))
                    label_pred = label_pred.squeeze()
                    labels = labels.squeeze()
                    loss = torch.nn.functional.cross_entropy(label_pred,labels)
                    lab_out = torch.argmax(label_pred,dim=1)
                    f1_temp = f_score(labels.cpu().numpy(),lab_out.cpu().numpy())
                    acc_temp = accuracy(labels.cpu().numpy(),lab_out.cpu().numpy())
                    if (np.isinf(loss.item()) or np.isnan(loss.item())):
                        print(f'Bad loss, skipping the batch {i}')
                        del loss,label_pred
                        gc_collect()
                        continue
                        
                    ## Training Code    
                    loss.backward()
                    optim.step()
                    scheduler.step(loss)
                    
                    lr = scheduler.get_last_lr()[0] if scheduler else lr
                    train_progress.set_description(f'Train: loss:{loss:.4f},lr:{lr:.4f},f1:{np.mean(f1_temp):.4f},acc:{acc_temp:.4f}')
                    
        if test_dl is not None:
            pred_val,targets_val,losses_val,f1_val,acc_val = evaluate_model(model,test_dl,device,'Eval')
            # pred_train,targets_train,losses_train,f1_train,acc_train = evaluate_model(model,train_dl,device,'Train_Eval')
            val_loss.append(np.mean(losses_val))
            # train_loss.append(np.mean(losses_train))
            val_fscore.append(np.mean(f1_val))
            # train_fscore.append(np.mean(f1_train))
            if (np.mean(f1_val)>best_eval_score):
                print(best_eval_score," ---> ",np.mean(f1_val))
                best_eval_score = np.mean(f1_val)
                torch.save(model.state_dict(), 'D:\\HSI Project\\Updated_Work\\HSI_Classification\\HybridSN\\HSI_Minerals\\hybrid_cnn_model_demo.pth')
                
            print("Per Epoch Test f1: ",f1_val)
            print("Per Epoch Accuracy: ",acc_val)
                
    if test_dl is not None:
        pred,targets,losses,f1,acc = evaluate_model(model,test_dl,device,'Eval')
        
        print("Final Test f1: ",f1)
        print("Final Accuracy: ",acc)
    return model,train_loss,val_loss,train_fscore,val_fscore
