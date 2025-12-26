import os
import argparse
import time

import torch
import streamlit as st

from models.definitions.transformer_net import TransformerNet
from utils.image import *

@st.cache_resource
def load_model(model_path, device):
    transformer_net = TransformerNet().to(device)
    
    # Load model state
    state_dict = torch.load(model_path)
    transformer_net.load_state_dict(state_dict, strict=True)
    
    transformer_net.eval().to(device)
    return transformer_net

def stylize(config, uploaded_image=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transformer_net = load_model(os.path.join(config["pretrained_models_path"], config["model_name"]), device)
    
    # Weight check
    # for k, v in state_dict.items():
    #     print(k, v.abs().mean())
    
    start_time = time.time()
    with torch.no_grad():
        if uploaded_image is None:
            content_image_path = os.path.join("./src/data/content-images", config["content_image"])
            content_image = prepare_image(content_image_path, device, config['image_size'], 1)
        
        content_image = prepare_image(uploaded_image, device, config['image_size'], 1)
        
        # feedforward
        stylized_image = transformer_net(content_image)

        if uploaded_image is None:
            save_image(stylized_image, config)
    
    print("Image Stylized")
    print(f"FeedForward time taken: {time.time() - start_time}")   
    
    return stylized_image     

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, help="Name of the pretrained model to load", default="style_model.pth")
    parser.add_argument("--content_image", type=str, help="Content image name", default=None, required=True)
    parser.add_argument("--output_path", type=str, help="Output image path", default="./src/data/output")
    parser.add_argument("--output_image", type=str, help="Output image name", required=True)

    args = parser.parse_args()
    
    config = dict()
    
    for arg in vars(args):
        config[arg] = getattr(args, arg)
        
    image_size = 256      
        
    config['image_size'] = image_size        
    config["pretrained_models_path"] = "./src/models/pretrained"
    
    stylize(config)        