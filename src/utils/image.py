import os
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
from streamlit.runtime.uploaded_file_manager import UploadedFile
import io

IMAGENET_MEAN = np.array([0.485,0.456,0.406])
IMAGENET_STD = np.array([0.229,0.224,0.225])

def get_transform(target_width):
    transform = transforms.Compose([
            transforms.Resize(target_width),
            transforms.CenterCrop(target_width),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)
        ])
    
    return transform 

# def load_image(image_path, target_shape=None):
    '''
        Input parameters: image_path, shape
        
        read the image
        convert BGR to RGB
        reshape to scale to shape
        convert to float
        normalize to [0,1]
    '''
    
    # Load image
    # image = cv.imread(image_path)
    
    # Convert from BGR to RGB
    # image = image[:,:,::-1]
    
    image = Image.open(image_path).convert("RGB")
    
    if isinstance(target_shape, int):
        # reshape to scale to the target_shape
        curr_width, curr_height = image.size
        new_width = target_shape
        new_height = int(curr_height * (new_width / curr_width))
        # resize the image
        image = image.resize((new_width, new_height), Image.LANCZOS)
    else:
        image = image.resize((target_shape[1], target_shape[0]), Image.LANCZOS)
    
        
    # image = image.astype(float)
    
    # standardize
    # image /= 255.0
    
    return image   
    

def prepare_image(input_source, device, shape, batch_size):
    # image = load_image(path, shape)
    
    if isinstance(input_source, str):
        image = Image.open(input_source).convert("RGB")
    elif isinstance(input_source, UploadedFile):
        input_source.seek(0)
        image_bytes = input_source.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    elif isinstance(input_source, Image.Image):
        image = input_source.convert("RGB")
    
    
    transform = get_transform(shape)
    image = transform(image).unsqueeze(0)

    if batch_size > 1:
        image = image.repeat(batch_size, 1, 1, 1)
    
    return image.to(device)

def detransform(image):
    image = image.squeeze(0)

    image = image.detach().cpu().float()
    
    mean = torch.tensor(IMAGENET_MEAN).view(3,1,1)
    std = torch.tensor(IMAGENET_STD).view(3,1,1)

    image = image * std + mean  
    
    image = torch.clamp(image, 0, 1)
    
    image = transforms.ToPILImage()(image)
    
    return image

def save_image(image, config):
    image = detransform(image)
    
    os.makedirs(config['output_path'], exist_ok=True)
    output_path = os.path.join(config["output_path"], config["output_image"])
    
    image.save(output_path)

    print("Stylized image saved at: ", output_path)
    
def PIL_to_bytes(image, format="PNG"):
    buffer = io.BytesIO()
    image.save(buffer, format=format)   
    buffer.seek(0)
    return buffer.getvalue()