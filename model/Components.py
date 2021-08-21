import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self,in_c,out_c,k = 3,s = 1,p = 1):
        super(ConvBlock,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size = k,stride = s, padding = p,bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU6(inplace=True)
        )
    
    def forward(self,x):
        return self.layers(x)

class DepthwiseSeparableConvBlock(nn.Module):
    def __init__(self,in_c,out_c,k=3,s=1,p=1):
        super(DepthwiseSeparableConvBlock,self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels = in_c,out_channels=in_c,kernel_size = k, stride = s,padding = p,groups = in_c,bias=False),
            nn.BatchNorm2d(in_c),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels = in_c,out_channels = out_c,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_c),
        )
    
    def forward(self,x):
        return self.layers(x)

class InvertedResidualBlock(nn.Module):
    def __init__(self,in_c,out_c,s,expansion_factor):
        super(InvertedResidualBlock,self).__init__()
        self.isAdded = (s == 1) and (in_c == out_c)
        hidden_dim = (int)(in_c*expansion_factor)
        if expansion_factor == 1:
            self.layers = DepthwiseSeparableConvBlock(in_c,out_c,k = 3,s = s)
        
        else:
            self.layers = nn.Sequential(
                ConvBlock(in_c,hidden_dim,k=1,s=1,p=0),
                DepthwiseSeparableConvBlock(hidden_dim,out_c,k=3,s=s,p=1)
            )
    
    def forward(self,x):
        if self.isAdded:
            return x+self.layers(x)
        
        return self.layers(x)

class ResidualBlock(nn.Module):
    def __init__(self,in_c,out_c,s=1):
        super(ResidualBlock,self).__init__()
        self.main_layers = nn.Sequential(
            DepthwiseSeparableConvBlock(in_c,out_c,k=3,s=s,p=1),
            ConvBlock(out_c,out_c,k=1,s=1,p=0),
            DepthwiseSeparableConvBlock(out_c,out_c,k=3,s=1,p=1),
            nn.Conv2d(out_c,out_c,kernel_size=1,stride=1,padding=0,bias=False),
            nn.BatchNorm2d(out_c)
        )
        if in_c == out_c:
            self.residual_layer = None
        else:
            self.residual_layer = nn.Sequential(
                nn.Conv2d(in_c,out_c,kernel_size=1,stride=1,padding=0,bias=False),
                nn.BatchNorm2d(out_c)
            )
        
        self.relu = nn.ReLU6(inplace=True)
    
    def forward(self,x):
        residual = x
        out = self.main_layers(x)
        if self.residual_layer is not None:
            residual = self.residual_layer(x)
        
        out += residual
        out = self.relu(out)
        return out

