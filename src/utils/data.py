from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
import os
from torchvision import transforms, datasets
from PIL import Image

from utils.image import *

IMAGENET_MEAN = np.array([0.485,0.456,0.406])
IMAGENET_STD = np.array([0.229,0.224,0.225])

class NSTDataset(Dataset):
    
    def __init__(self, image_dir, target_width):
        '''
            Input parameters: image_dir_path, width

            load image
            transform(tensor, normalize)
        '''
        
        self.image_dir = image_dir
        self.target_width = target_width
        
        # List of all image paths
        self.img_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
        
        # image = load_image(self.img_paths[0], self.target_width)
        # h, w = image.size[::-1]
        
        # self.target_height = int(h * (target_width / w))
        
        self.transform = get_transform(self.target_width)
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image = Image.open(self.img_paths[index]).convert("RGB")
        image = self.transform(image)
        
        return image, 0

class SubsetSampler(Sampler):
    def __init__(self, dataset, subset_size):
        self.dataset = dataset
        
        if subset_size is None:
            subset_size = len(self.dataset)
        
        assert 0 < subset_size <= len(dataset), f"Subset size should be between [0, {len(dataset)}]"
        self.subset_size = subset_size
    
    def __len__(self):
        return self.subset_size
    
    def __iter__(self):
        return iter(range(self.subset_size))
                    
                    

def get_data_loader(config):
    '''
        load dataset from datasets.ImageFolder
        provide transform object
        initialize loader
        return loader
    '''

    # ImageFolder requires the images to be stored in class folders, which is not required in NST
    # dataset = datasets.ImageFolder(config['dataset_path'], transform)
    
    dataset = NSTDataset(config['dataset_path'], config['image_size'])

    sampler = SubsetSampler(dataset, config['subset_size'])
    config['subset_size'] = sampler.subset_size
    
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=config['batch_size'], drop_last=True)
    
    return dataloader