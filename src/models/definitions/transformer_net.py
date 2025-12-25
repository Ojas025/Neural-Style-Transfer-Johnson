import torch
import torch.nn as nn

class TransformerNet(nn.Module):
    '''
        TransformerNet Architecture
        
        Input   3x256x26
        Conv(stride=1, kernel=9x9)  32x256x256
        Conv(stride=2, kernel=3x3)  64x128x128
        Conv(stride=2, kernerl=3x3) 128x64x64
        ResidualBlock(filters=128)  128x64x64
        ResidualBlock(filters=128)  128x64x64
        ResidualBlock(filters=128)  128x64x64
        ResidualBlock(filters=128)  128x64x64
        ResidualBlock(filters=128)  128x64x64
        Conv(stride=0.5, kernel=3x3)    64x128x128
        Conv(stride=0.5, kernel=3x3)    32x256x256
        Conv(stride=1, kernel=9x9)  3x256x256
    '''
    
    def __init__(self):
        super().__init__()
        
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
        # Down sampling convolution layers
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4)
        self.instance1 = nn.InstanceNorm2d(num_features=32, affine=True)

        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.instance2 = nn.InstanceNorm2d(num_features=64, affine=True)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.instance3 = nn.InstanceNorm2d(num_features=128, affine=True)

        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        
        # Up sampling convolution layers
        self.upsample1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv4 = nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.instance4 = nn.InstanceNorm2d(num_features=64, affine=True)
        
        self.upsample2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.instance5 = nn.InstanceNorm2d(num_features=32, affine=True)
        
        self.conv6 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=9, stride=1, padding=4)
    
    def forward(self, x):
        x = self.relu(self.instance1(self.conv1(x)))
        x = self.relu(self.instance2(self.conv2(x)))
        x = self.relu(self.instance3(self.conv3(x)))
        
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        x = self.res5(x)
        
        x = self.upsample1(x)
        x = self.relu(self.instance4(self.conv4(x)))

        x = self.upsample2(x)
        x = self.relu(self.instance5(self.conv5(x)))

        # x = self.tanh(self.conv6(x))
        x = self.conv6(x)
        
        return x
        
    
class ResidualBlock(nn.Module):
    
    def __init__(self, channels):
        super().__init__()
        
        self.channels = channels
        
        self.relu = nn.ReLU()

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.instance1 = nn.InstanceNorm2d(num_features=channels, affine=True)
        
        self.conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1)
        self.instance2 = nn.InstanceNorm2d(num_features=channels, affine=True)
    
    def forward(self, x):
        identity = x
        
        x = self.relu(self.instance1(self.conv1(x)))
        x = self.instance2(self.conv2(x))
        
        x += identity
        return self.relu(x)