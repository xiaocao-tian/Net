import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, inchannle=3, outchannel=32, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.d_s_conv = nn.Sequential(
            nn.Conv2d(inchannle, inchannle, kernel_size=3, stride=stride, padding=1, groups=inchannle),
            nn.BatchNorm2d(inchannle),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(inchannle, outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.d_s_conv(x)
        


class MobileNetV1(nn.Module):
    def __init__(self, inchannle=1, classes=10):
        super(MobileNetV1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannle, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.depthwise_separable_conv = nn.Sequential(
            DepthwiseSeparableConv(inchannle=32,   outchannel=64,   stride=1),
            DepthwiseSeparableConv(inchannle=64,   outchannel=128,  stride=1),
            DepthwiseSeparableConv(inchannle=128,  outchannel=128,  stride=1),
            DepthwiseSeparableConv(inchannle=128,  outchannel=256,  stride=1),
            
            DepthwiseSeparableConv(inchannle=256,  outchannel=256,  stride=1),
            DepthwiseSeparableConv(inchannle=256,  outchannel=512,  stride=1),
            
            DepthwiseSeparableConv(inchannle=512,  outchannel=512,  stride=1),
            DepthwiseSeparableConv(inchannle=512,  outchannel=512,  stride=1),
            DepthwiseSeparableConv(inchannle=512,  outchannel=512,  stride=1),
            DepthwiseSeparableConv(inchannle=512,  outchannel=512,  stride=1),
            DepthwiseSeparableConv(inchannle=512,  outchannel=512,  stride=1),
            
            DepthwiseSeparableConv(inchannle=512,  outchannel=1024, stride=2),
            DepthwiseSeparableConv(inchannle=1024, outchannel=1024, stride=1)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, classes)
        
    def forward(self, x):
        out = self.conv(x)
        out = self.depthwise_separable_conv(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out


def loadModel(model):
    if model == "MobileNetV1":
        model = MobileNetV1(inchannle=1)
    return model
