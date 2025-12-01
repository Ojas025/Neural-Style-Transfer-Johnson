import torch
import torch.nn as nn
from torchvision import models

class VGG16(nn.Module):
    
    def __init__(self):
        super().__init__()

        self.vgg = models.vgg16(pretrained=True).features
        
        self.selected_layers = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }
        
        for param in self.parameters():
            param.requires_grad_(False)
        
    
    def forward(self, x):
        feature_maps = {}
        
        for index, layer in enumerate(self.vgg):
            x = layer(x)
            
            if index in self.selected_layers.items():
                for name, i in {**self.selected_layers}.items():
                    if index == i:
                        feature_maps[name] = x
        
        return feature_maps                        

PerceptualLossNetwork = VGG16