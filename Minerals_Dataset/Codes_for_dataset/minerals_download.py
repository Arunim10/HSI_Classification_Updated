# import os
# import requests
# from tqdm import tqdm

# file_urls = []

# for id in range(130):
#     file = f'https://zenodo.org/records/1476503/files/{id:04d}.zip?download=1'
#     file_urls.append(file)
    
# # print(file_urls)

# save_dir = "D:/HSI Project/Updated_Work/HSI_Classification/Minerals_Dataset/Masked_HRD_Data/"

# for url in file_urls:
#     filename = url.split('/')[-1][:4] + ".zip"
    
#     if os.path.exists(os.path.join(save_dir, filename)):
#         print(f"{filename} already exists. Skipping download.")
#         continue
    
#     response = requests.head(url)
#     file_size = int(response.headers.get('content-length', 0))

#     # Initialize tqdm with the total file size
#     progress_bar = tqdm(total=file_size, unit='B', unit_scale=True, desc=filename, ascii=True)

#     # Start downloading the file
#     response = requests.get(url, stream=True)
#     with open(os.path.join(save_dir, filename), 'wb') as f:
#         for data in response.iter_content(chunk_size=1024):
#             f.write(data)
#             progress_bar.update(len(data))

#     # Close the tqdm progress bar
#     progress_bar.close()

#     print(f"Downloaded {filename}")

# print("All files downloaded successfully.")

# # print(file_urls[0].split('/')[-1][:4])

# from glob import glob
# import zipfile
# import os
# from tqdm import tqdm

# hdr_zip_paths = glob(r"D:\HSI Project\Updated_Work\HSI_Classification\Minerals_Dataset\HDR_Data\*")
# hdr_extract_folder_path = f"D:/HSI Project/Updated_Work/HSI_Classification/Minerals_Dataset/HDR_Data/Extracted_Folder/"

# mask_zip_paths = glob(r"D:\HSI Project\Updated_Work\HSI_Classification\Minerals_Dataset\Masked_HRD_Data\*")
# mask_extract_folder_path = f"D:/HSI Project/Updated_Work/HSI_Classification/Minerals_Dataset/Masked_HRD_Data/Extracted_Folder"

# # Create the extraction folder if it doesn't exist
# os.makedirs(mask_extract_folder_path, exist_ok=True)


# for zip_file_path in tqdm(mask_zip_paths,total=len(mask_zip_paths)):
#     # Open the zip file
#     with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
#         # Extract all the contents into the extraction folder
#         zip_ref.extractall(mask_extract_folder_path)

# print("Extraction complete.")

