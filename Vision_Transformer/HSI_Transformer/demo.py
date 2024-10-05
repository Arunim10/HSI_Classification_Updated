from glob import glob
from scipy.io import loadmat
import numpy as np
# cache = {}
paths = glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\*.mat")
hsi_paths = []
for path in paths:
    name = path.split("\\")[-1]
    if "gt" not in name:
        hsi_paths.append(path)
print(len(hsi_paths))
print(len(np.unique(np.array(hsi_paths))))
cache = {}
for path in hsi_paths:
    cache[path] =  np.transpose(loadmat(path)['HDR'][:],axes=(1,2,0))
    
print(len(cache))