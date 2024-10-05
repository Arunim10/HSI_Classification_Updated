import pandas as pd
from glob import glob
import re
from tqdm import tqdm

def Mapping_MatFiles_to_Minerals():
    paths = glob("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_mat_files\\*.mat")

    with open('D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Mineral_IDs.txt', 'r') as file:
        text = file.read()

    # Split the text into lines and then into columns
    lines = text.strip().split('\n')
    data = [line.split() for line in lines]

    # id experiment(A/B) mineral hsi_path mask_path
    df = pd.DataFrame(columns=['ID','Experiment','Mineral','HSI_Path','MASK_Path'])
    # print(df.columns)
    for i in tqdm(range(1,len(data)),total=len(data)-1):
        mineral = data[i][0]
        datapoints = data[i][-1]
        ids = []
        for j in range(1,len(data[i])-1):
            id = re.sub(r'\(2\)|,', '', data[i][j])  # remove '(2)' and ','
            ids.append(id)
            
        for id in ids:
            id_path = sorted([path for path in paths if id in path])
            if len(id_path)==2:
                row = {'ID':id,'Experiment':'A','Mineral':mineral,'HSI_Path':id_path[0],'MASK_Path':id_path[1]}
                df.loc[len(df)] = row
            elif len(id_path)==4:
                row1 = {'ID':id,'Experiment':'A','Mineral':mineral,'HSI_Path':id_path[0],'MASK_Path':id_path[1]}
                row2 = {'ID':id,'Experiment':'B','Mineral':mineral,'HSI_Path':id_path[2],'MASK_Path':id_path[3]}
                df.loc[len(df)] = row1
                df.loc[len(df)] = row2
                
    df.to_csv("D:\\HSI Project\\Updated_Work\\HSI_Classification\\Minerals_Dataset\\Minerals_Mapped.csv",index=False)    
    
    print("Minerals Mapped Dataframe successfully created and saved successfully !!")    
if __name__=="__main__":
    Mapping_MatFiles_to_Minerals()


            
     
     
    
    