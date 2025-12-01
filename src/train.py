import os
import argparse
import time

import torch
from torch.optim import Adam

from models.definitions.perceptual_loss_network import PerceptualLossNetwork
from models.definitions.transformer_net import TransformerNet
from utils.data import *
from utils.image import *
from utils.losses import *
from utils.utils import *


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # INITIALIZE NETWORKS
    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNetwork().eval().to(device)
    
    dataloader = get_data_loader(config)
    
    optimizer = Adam(transformer_net.parameters(), lr=config['learning_rate'])
    
    # FETCH STYLE IMAGE
    style_image = prepare_image(style_image_path, device=device)
    
    style_feature_maps = perceptual_loss_net(style_image)
    
    style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    content_layer = "relu2_2"
    
    # COMPUTE GRAM MATRICES FOR STYLE IMAGE
    target_style_representations = [
        gram_matrix(style_feature_maps[layer])
        for layer in style_layers
    ]
    
    start_time = time.time()
    for iteration in range(config['num_iterations']):
        for batch, (content_batch, _) in enumerate(dataloader):
            optimizer.zero_grad()
            content_batch = content_batch.to(device)
            
            # feedforward through transformer net
            current_batch = transformer_net(content_batch)
            
            # get feature maps through perceptual net
            content_feature_maps = perceptual_loss_net(content_batch)
            current_feature_maps = perceptual_loss_net(current_batch)
            
            target_content_representation = content_feature_maps[content_layer]
            current_content_representation = current_feature_maps[content_layer]
            content_loss = torch.nn.functional.mse_loss(target_content_representation, current_content_representation)
            
            # calculate gram matrix for current_batch
            current_style_representation = [
                gram_matrix(current_feature_maps[layer])
                for layer in style_layers
                
            ]
            
            style_loss = 0.0
            for gram_target, gram_current in zip(target_style_representations, current_style_representation):
                style_loss += torch.nn.functional.mse_loss(gram_target, gram_current)
            
            style_loss /= len(current_style_representation)
            
            # calculate total variation loss
            total_variation_loss = compute_total_variation_loss(current_batch)
            
            # calculate total loss
            total_loss = (config['content_weight'] * content_loss) + (config['style_weight'] * style_loss) + (config['total_variation_weight'] * total_variation_loss)
            
            # compute gradients
            total_loss.backward()

            # update weights
            optimizer.step()
            
            if batch % 50 == 0:
                print(f"Epoch {iteration}, Batch {batch} | "
                f"Total Loss: {total_loss.item():.4f} | "
                f"Content: {config['content_weight'] * content_loss.item():.2f} | " 
                f"Style: {config['style_weight'] * style_loss.item():.2f} | "
                f"Total Variation: {config['total_variation_weight'] * total_variation_loss.item():.2f}"
                )
            
    pretrained_model_path = "./data/pretrained/style_model.pth"
    os.makedirs(os.path.dirname(pretrained_model_path), exist_ok=True)
    
    torch.save(transformer_net.state_dict(), pretrained_model_path)
    print("Model saved") 
    print(f"Training time required: {time.time() - start_time}")           
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--style_image', type=str, help='Style Image Name', default='starry_night.jpg')
    parser.add_argument('--content_weight', type=float, help='Weight factor for content loss', default=1e0)
    parser.add_argument('--style_weight', type=float, help='Weight factor for style loss', default=1e5)
    parser.add_argument('--total_variation_weight', type=float, help='Weight factor for total variation loss', default=0)
    parser.add_argument('--num_iterations', type=int, help='Number of training iterations', default=2)
    parser.add_argument('--subset_size', type=int, help='Subset size to use from MS COCO dataset', default=None)
    parser.add_argument('--learning_rate', type=float, help='learning rate for adam optimizer', default=1e-3)
    
    args = parser.parse_args()


    config = dict()

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    batch_size = 4
    image_size = 256
    
    dataset_path = os.path.join(".", "data", "mscoco", "train2014")
    style_image_path = os.path.join(".", "data", "style-images", config["style_image"])
    
    config['dataset_path'] = dataset_path
    config['batch_size'] = batch_size
    config['image_size'] = image_size
    
    train(config)