import cv2 as cv
import os
from PIL import Image
from torchvision import transforms

def load_image(image_path, target_shape=None):
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
        curr_height, curr_width = image.size
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
    

def prepare_image(path, device, shape, batch_size):
    image = load_image(path, shape)
    
    tensor_transformation = transforms.ToTensor()
    image = tensor_transformation(image).squeeze(0) 
    image = image.to(device)
    
    image = image.repeat(batch_size, 1, 1, 1)
    
    return image

def save_image(image, config):
    os.makedirs(config['output_path'], exist_ok=True)
    output_path = os.path.join(config["output_path"], config["output_image"])
    
    cv.imwrite(output_path, image)
    print("Stylized image saved at: ", output_path)
