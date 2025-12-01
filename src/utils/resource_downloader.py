import zipfile
import os
from torch.hub import download_url_to_file

MS_COCO_TRAIN_DATASET_URL = r'http://images.cocodataset.org/zips/train2014.zip'

resource_tmp_path = 'train.zip'

# Download the dataset
print(f"Downloading MS COCO 2014 train set from {MS_COCO_TRAIN_DATASET_URL}")
download_url_to_file(MS_COCO_TRAIN_DATASET_URL, resource_tmp_path)

# Unzip 
local_dataset_path = os.path.join("..", "data", "mscoco")
os.makedirs(local_dataset_path, exist_ok=True)

print(f"Unzipping to {local_dataset_path}")

with zipfile.ZipFile(resource_tmp_path, "r") as z:
    z.extractall(path=local_dataset_path)

# Remove temp zip
os.remove(resource_tmp_path) 
print("Resource downloaded and extracted successfully")