import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.cuda.amp import autocast
from flask import Flask, request, jsonify, render_template
from model import HSI_Model
import base64
from io import BytesIO
import warnings
warnings.filterwarnings("ignore")
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')
 
def image_preprocess(hf_iron_test):
    iron_test = np.array(hf_iron_test.get('hdr')).copy()
    # iron_test[0,253,183] = iron_test[1,253,183]
    if np.isnan(iron_test[0,:,:]).any():
        for x,y in zip(np.where(np.isnan(iron_test[0,:,:]))[0],np.where(np.isnan(iron_test[0,:,:]))[1]):
            print("Nan value detected!")
            iron_test[0,x,y] = iron_test[1,x,y]
    cropped_data_test = []
    for i in range(256):
        cropped_data_test.append(iron_test[i,:,:])

    for i in range(1,256):
        if (np.isnan(cropped_data_test[i]).any()):
            img = cropped_data_test[i]
            for (nan_x,nan_y) in zip(np.where(np.isnan(img))[0],np.where(np.isnan(img))[1]):
                print("Nan value detected!")
                img[nan_x,nan_y] = cropped_data_test[i-1][nan_x,nan_y]

    iron_arr_test = np.stack(cropped_data_test)
    k = 7
    x = torch.tensor(np.array(iron_arr_test.copy(),dtype=np.float32))
    x = x.unsqueeze(0)
    padding = [k//2,k//2,k//2,k//2]
    padd = F.pad(x,padding)
    # print(padd.shape)
    data_test = {}
    data_test["test"] = []
    for i in range(iron_arr_test.shape[1]):
        for j in range(iron_arr_test.shape[2]):
            temp = padd[:,:,i:i+k//2+k//2+1,j:j+k//2+k//2+1]
            data_test["test"].append(temp)

    def convert_test(dicti):
        X = []
        for labels in dicti:
            for ten in dicti[labels]:
                X.append(ten)
        return X

    test_X = convert_test(data_test)
    
    return test_X, iron_arr_test

class BuildDatasetTest(torch.utils.data.Dataset):
    def __init__(self,data):
        super().__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,idx):
        spectral_values = self.data[idx].squeeze(0)
        # mean = torch.mean(spectral_values)
        # std = torch.std(spectral_values)
        # spectral_values = (spectral_values-mean)/std
        return torch.tensor(spectral_values,dtype=torch.float32)

def evaluate_test(model,dl,device):
    torch.manual_seed(42)
    model = model.to(device)
    pred = []
    with torch.no_grad():
        model.eval()
        with tqdm(dl,desc='Eval',mininterval=30,total=len(dl)) as progress:
            for i,spec_values in enumerate(progress):
                with autocast(enabled=True):
                # if True:
                    spec_values = spec_values.to(device,dtype=torch.float32)
                    label_pred = model(spec_values).squeeze(1)
                    lab_out = torch.argmax(label_pred,dim=1)
                    pred.append(lab_out.cpu().numpy())
    pred = np.concatenate(pred)
    return pred

def visualization(prediction,hdr_array):
    from PIL import Image
    color_image_np = np.stack((hdr_array,) * 3, axis=-1)
    colors = [
        [0, 0, 0],   # Black # Background
        [0, 255, 0],   # Green # Biotite
        [0, 0, 255],   # Blue # Calcite
        [255, 255, 0], # Yellow # Copper
        [255, 0, 255], # Magenta # Garnet
        [0, 255, 255],  # Cyan 3 Quartz
        [128, 128, 0],   # Olive # Celestite
        [0, 128, 128],   # Teal # Dolomite
        [128, 0, 128],   # Purple # Graphite
        [255, 165, 0],   # Orange # Olivine
        [255, 140, 105]   # Salmon # Zircon
    ]
    for i in range(len(prediction)):
        row = i//410
        col = i%410
        color_image_np[row,col,:] = colors[prediction[i]]
    
    image_pil = Image.fromarray(color_image_np.astype(np.uint8))
    image_pil.save('output_image.png')
    return image_pil


@app.route('/predict', methods=['POST'])
def predict(hf_iron_test=h5py.File('data.h5','r')):

    test_X, iron_arr_test = image_preprocess(hf_iron_test)
    test_ds = BuildDatasetTest(test_X)
    test_dataloader = DataLoader(test_ds,batch_size=2048,num_workers=4,shuffle=False,pin_memory=True,drop_last = False)
    prediction = evaluate_test(model,test_dataloader,device)

    display_pred = iron_arr_test[0,:,:].copy()

    visualization(prediction, display_pred)
  
    # Read the content of the PNG file
    with open('output_image.png', 'rb') as image_file:
        image_data = image_file.read()

    # Encode the image data to Base64
    base64_encoded = base64.b64encode(image_data).decode()
    
    # return jsonify({'output_image': base64_encoded})

if __name__ == '__main__':
    weight_path = 'Model_HSI.pth'
    model = HSI_Model(256,128,64,32,11)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model.load_state_dict(torch.load(weight_path))
    model.eval()
    predict(h5py.File('data.h5','r'))

# if __name__ == '__main__':
#     weight_path = 'Model_HSI.pth'
#     model = HSI_Model(256,128,64,32,11)
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     model.load_state_dict(torch.load(weight_path))
#     model.eval()
#     app.run(debug=True, port=8000)