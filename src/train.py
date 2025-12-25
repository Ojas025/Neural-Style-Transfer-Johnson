import os
import argparse
import time
import warnings

import torch
from torch.optim import Adam

from models.definitions.perceptual_loss_network import PerceptualLossNetwork
from models.definitions.transformer_net import TransformerNet
from utils.data import *
from utils.image import *
from utils.losses import *
from utils.utils import *

warnings.filterwarnings('ignore', category=UserWarning)


def train(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # INITIALIZE NETWORKS
    transformer_net = TransformerNet().train().to(device)
    perceptual_loss_net = PerceptualLossNetwork().eval().to(device)
    
    dataloader = get_data_loader(config)
    
    optimizer = Adam(transformer_net.parameters(), lr=config['learning_rate'])
    
    # FETCH STYLE IMAGE
    style_image = prepare_image(style_image_path, device, config['image_size'], config['batch_size'])
    
    with torch.no_grad():
        style_feature_maps = perceptual_loss_net(style_image)
    
    style_layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3"]
    content_layer = "relu2_2"
    
    # COMPUTE GRAM MATRICES FOR STYLE IMAGE
    target_style_representations = [
        gram_matrix(style_feature_maps[layer]).detach()
        for layer in style_layers
    ]
    
    # for g in target_style_representations:
    #     print("target gram mean/std:", g.mean().item(), g.std().item())

    preview_image = prepare_image('./src/data/content-images/lion.jpg', device, shape=config['image_size'], batch_size=1)

    
    start_time = time.time()
    
    try:
        for iteration in range(config['num_epochs']):
            for batch, (content_batch, _) in enumerate(dataloader):
                optimizer.zero_grad()
                content_batch = content_batch.to(device)
                
                # feedforward through transformer net
                current_batch = transformer_net(content_batch)
                
                # get feature maps through perceptual net
                with torch.no_grad():
                    content_feature_maps = perceptual_loss_net(content_batch)
                    
                current_feature_maps = perceptual_loss_net(current_batch)
                
                # get content loss
                content_loss = compute_content_loss(content_feature_maps, current_feature_maps, content_layer)
                
                # get style loss
                style_loss = compute_style_loss(current_feature_maps, style_layers, target_style_representations)
                
                # calculate total variation loss
                total_variation_loss = compute_total_variation_loss(current_batch)
                
                # print("Content Loss: ", content_loss)
                # print("Style Loss: ", style_loss)
                # print("Total Variation Loss: ", total_variation_loss)
                
                # calculate total loss
                total_loss = (config['content_weight'] * content_loss) + (config['style_weight'] * style_loss) + (config['total_variation_weight'] * total_variation_loss)
                
                # compute gradients
                total_loss.backward()

                # update weights
                optimizer.step()
                
                # if batch % 200 == 0:
                #     total_grad = sum(
                #         p.grad.abs().mean().item()
                #         for p in transformer_net.parameters()
                #         if p.grad is not None
                #     )
                #     print("Grad magnitude:", total_grad)
                #     print()
                
                if batch % 500 == 0:
                    with torch.no_grad():
                        transformer_net.eval()
                        preview_output = transformer_net(preview_image)
                        transformer_net.train()
                    
                    save_image(preview_output, { 'output_path': './src/data/previews', 'output_image': f'epoch{iteration}_batch{batch}.jpg' })                    
                
                if batch % 50 == 0:
                    print(f"Epoch {iteration}, Batch {batch} | "
                    f"Total Loss: {total_loss.item():.4f} | "
                    f"Content: {content_loss.item():.2f} | " 
                    f"Style: {style_loss.item():.4f} | "
                    f"Total Variation: {total_variation_loss.item():.2f}"
                    )
                
        pretrained_model_path = "./src/data/pretrained/style_model.pth"
        os.makedirs(os.path.dirname(pretrained_model_path), exist_ok=True)
        
        torch.save(transformer_net.state_dict(), pretrained_model_path)
        print("Model saved") 
        print(f"Training time required: {time.time() - start_time}")           
    
    except KeyboardInterrupt:
        print("Keyboard interruption. Saving model...")

        pretrained_model_path = "./src/data/pretrained/style_model.pth"
        os.makedirs(os.path.dirname(pretrained_model_path), exist_ok=True)
        
        torch.save(transformer_net.state_dict(), pretrained_model_path)
        print("Model saved") 
        print(f"Training time required: {time.time() - start_time}")
        return  

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--style_image', type=str, help='Style Image Name', default='starry_night.jpg')
    parser.add_argument('--content_weight', type=float, help='Weight factor for content loss', default=1e0)
    parser.add_argument('--style_weight', type=float, help='Weight factor for style loss', default=6e5)
    parser.add_argument('--total_variation_weight', type=float, help='Weight factor for total variation loss', default=1e-6)
    parser.add_argument('--num_epochs', type=int, help='Number of training epochs', default=2)
    parser.add_argument('--subset_size', type=int, help='Subset size to use from MS COCO dataset', default=None)
    parser.add_argument('--learning_rate', type=float, help='learning rate for adam optimizer', default=1e-3)
    
    args = parser.parse_args()


    config = dict()

    for arg in vars(args):
        config[arg] = getattr(args, arg)

    batch_size = 4
    image_size = 256
    
    # dataset_path = os.path.join(".", "data", "mscoco", "train2014")
    dataset_path = "./src/data/mscoco/train2014"
    # os.makedirs(dataset_path, exist_ok=True)
    style_image_path = os.path.join(".", "src", "data", "style-images", config["style_image"])
    
    config['dataset_path'] = dataset_path
    config['batch_size'] = batch_size
    config['image_size'] = image_size
    
    train(config)