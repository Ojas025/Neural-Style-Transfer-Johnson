import torch

from utils.utils import *

def compute_total_variation_loss(feature_maps):
    batch_size = feature_maps.shape[0]
    
    horizontal_difference = torch.abs(feature_maps[:,:, :, 1:] - feature_maps[:,:,:,:-1])
    
    vertical_difference = torch.abs(feature_maps[:,:,1:,:] - feature_maps[:,:,:-1,:])
    
    return (torch.sum(horizontal_difference) + torch.sum(vertical_difference)) / batch_size

def compute_style_loss(current_feature_maps, style_layers, target_style_representations):
    # calculate gram matrix for current_batch
    current_style_representation = [
        gram_matrix(current_feature_maps[layer])
        for layer in style_layers
        
    ]
    
    style_loss = 0.0
    
    for layer, gram_target in zip(style_layers, target_style_representations):
        gram_current = gram_matrix(current_feature_maps[layer])
        style_loss += torch.nn.functional.mse_loss(gram_current, gram_target)
        
    # print(style_loss)        
    
    style_loss /= len(style_layers)
    return style_loss

def compute_content_loss(content_feature_maps, current_feature_maps, content_layer):
    target_content_representation = content_feature_maps[content_layer]
    current_content_representation = current_feature_maps[content_layer]

    content_loss = torch.nn.functional.mse_loss(target_content_representation, current_content_representation)
    
    return content_loss