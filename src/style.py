import os
import argparse
import time

import torch

from models.definitions.transformer_net import TransformerNet
from utils.image import *

def stylize(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformer_net = TransformerNet().to(device)
    
    # Load model state
    state_dict = torch.load(os.path.join(config["pretrained_models_path"], config["model_name"]))
    transformer_net.load_state_dict(state_dict, strict=True)
    
    transformer_net.eval().to(device)
    
    start_time = time.time()
    with torch.no_grad():
        content_image_path = os.path.join("./data/content-images", config["content_image"])
        content_image = prepare_image(content_image_path, device)
        
        # feedforward
        stylized_image = transformer_net(content_image)

        output_path = os.path.join(config["output_path"], config["output_image"])
        save_image(stylized_image, output_path)
    
    print("Image Stylized")
    print(f"FeedForward time taken: {time.time() - start_time}")        

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, help="Name of the pretrained model to load", default="style_model.pth")
    parser.add_argument("--content_image", type=str, help="Content image name", default=None, required=True)
    parser.add_argument("--output_path", type=str, help="Output image path", default="./data/output")
    parser.add_argument("--output_image", type=str, help="Output image name", required=True)

    args = parser.parse_args()
    
    config = dict()
    
    for arg in vars(args):
        config[arg] = getattr(args, arg)
        
    config["pretrained_models_path"] = "./data/pretrained"
    
    stylize(config)        