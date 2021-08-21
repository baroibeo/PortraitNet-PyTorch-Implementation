import torch
import torch.nn as nn
from Components import*

class PortraitNet(nn.Module):
    def __init__(self,num_classes = 2):
        super(PortraitNet,self).__init__()
        self.first_conv = nn.Conv2d(3,32,kernel_size=1,stride=1,padding=0)

        #/1
        self.stage_1 = InvertedResidualBlock(32,16,1,1)

        #/2
        self.stage_2 = nn.Sequential(
            InvertedResidualBlock(16,24,2,6),
            InvertedResidualBlock(24,24,1,6),
        )

        #/4
        self.stage_3 = nn.Sequential(
            InvertedResidualBlock(24,32,2,6),
            InvertedResidualBlock(32,32,1,6),
            InvertedResidualBlock(32,32,1,6),
        )

        #/8
        self.stage_4 = nn.Sequential(
            InvertedResidualBlock(32,64,2,6),
            InvertedResidualBlock(64,64,1,6),
            InvertedResidualBlock(64,64,1,6),
            InvertedResidualBlock(64,64,1,6),
        )

        #/16
        self.stage_5 = nn.Sequential(
            InvertedResidualBlock(64,96,2,6),
            InvertedResidualBlock(96,96,1,6),
            InvertedResidualBlock(96,96,1,6),
        )

        #/32
        self.stage_6 = nn.Sequential(
            InvertedResidualBlock(96,160,2,6),
            InvertedResidualBlock(160,160,1,6),
            InvertedResidualBlock(160,160,1,6),
        )

        #/32
        self.stage_7 = InvertedResidualBlock(160,320,1,6)

        #Deconv
        self.deconv1 = nn.ConvTranspose2d(96,96,kernel_size=4,stride=2,padding=1,bias=False)
        self.deconv2 = nn.ConvTranspose2d(64,64,kernel_size=4,stride=2,padding=1,bias=False)
        self.deconv3 = nn.ConvTranspose2d(32,32,kernel_size=4,stride=2,padding=1,bias=False)
        self.deconv4 = nn.ConvTranspose2d(24,24,kernel_size=4,stride=2,padding=1,bias=False)
        self.deconv5 = nn.ConvTranspose2d(16,16,kernel_size=4,stride=2,padding=1,bias=False)

        self.dblock1 = ResidualBlock(320,96)
        self.dblock2 = ResidualBlock(96,64)
        self.dblock3 = ResidualBlock(64,32)
        self.dblock4 = ResidualBlock(32,24)
        self.dblock5 = ResidualBlock(24,16)

        #pred conv
        self.pred = nn.Conv2d(16,num_classes,kernel_size=3,stride=1,padding=1,bias=False)
    
    def forward(self,x):
        x = self.first_conv(x)
        encode_1_1 = self.stage_1(x)
        encode_1_2 = self.stage_2(encode_1_1)
        encode_1_4 = self.stage_3(encode_1_2)
        encode_1_8 = self.stage_4(encode_1_4)
        encode_1_16 = self.stage_5(encode_1_8)
        encode_1_32 = self.stage_6(encode_1_16)
        encode_1_32 = self.stage_7(encode_1_32)
        
        up_1_16 = self.deconv1(self.dblock1(encode_1_32)) # 96 x 14 x 14
        up_1_8 = self.deconv2(self.dblock2(up_1_16+encode_1_16)) # 64 x 28 x 28
        up_1_4 = self.deconv3(self.dblock3(up_1_8+encode_1_8)) # 32 x 56 x 56
        up_1_2 = self.deconv4(self.dblock4(up_1_4+encode_1_4)) # 24 x 112 x 112
        up_1_1 = self.deconv5(self.dblock5(up_1_2+encode_1_2)) # 16 x 224 x 224
        
        return self.pred(up_1_1)

def test():
    x = torch.randn(1,3,224,224)
    net = PortraitNet(2)
    print(net(x).shape)

if __name__ == '__main__':
    test()