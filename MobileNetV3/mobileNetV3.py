import torch.nn as nn


class SeModel(nn.Module):
    def __init__(self, inchannels=1, reduction=4):
        super(SeModel, self).__init__()
        self.SE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inchannels, inchannels//reduction, kernel_size=1),
            nn.BatchNorm2d(inchannels//reduction),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(inchannels//reduction, inchannels, kernel_size=1),
            nn.BatchNorm2d(inchannels),
            nn.Hardsigmoid(inplace=True)
        )
    def forward(self, x):
        return x * self.SE(x)


class bneck(nn.Module):
    def __init__(self, inchannels, exp, outchannels, ks, SEModel, NL, s):
        super(bneck, self).__init__()
        
        self.se = SEModel
        self.stride = s
        self.inchannels = inchannels
        self.outchannels = outchannels
        
        self.conv1 = nn.Conv2d(inchannels, exp, kernel_size=1, stride=1)
        self.bn1 = nn.BatchNorm2d(exp)
        self.nl = nn.ReLU(inplace=True) if NL=="RE" else nn.Hardswish(inplace=True)
        
        self.conv2 = nn.Conv2d(exp, exp, kernel_size=ks, stride=s, padding=ks//2, groups=exp)
        self.bn2 = nn.BatchNorm2d(exp)

        self.conv3 = nn.Conv2d(exp, outchannels, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(outchannels)        


    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.nl(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.nl(out)
        out = self.conv3(out)
        out = self.bn3(out)

        if self.se:
            out = self.se(out)

        if self.stride == 1 and self.inchannels == self.outchannels:
            out = out + x

        return out

class MobileNetV3Large(nn.Module):
    def __init__(self, inchannels=1, classes=1000):
        super(MobileNetV3Large, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish(inplace=True)
        )
        
        self.bneck = nn.Sequential(
            bneck(inchannels=16,  exp=16,  outchannels=16,  ks=3, SEModel=None,         NL="RE", s=1),
            bneck(inchannels=16,  exp=64,  outchannels=24,  ks=3, SEModel=None,         NL="RE", s=1),
            bneck(inchannels=24,  exp=72,  outchannels=24,  ks=3, SEModel=None,         NL="RE", s=1),
            bneck(inchannels=24,  exp=72,  outchannels=40,  ks=5, SEModel=SeModel(40),  NL="RE", s=1),
            bneck(inchannels=40,  exp=120, outchannels=40,  ks=5, SEModel=SeModel(40),  NL="RE", s=1),
            bneck(inchannels=40,  exp=120, outchannels=40,  ks=5, SEModel=SeModel(40),  NL="RE", s=1),
            bneck(inchannels=40,  exp=240, outchannels=80,  ks=3, SEModel=None,         NL="HS", s=1),
            bneck(inchannels=80,  exp=200, outchannels=80,  ks=3, SEModel=None,         NL="HS", s=1),
            bneck(inchannels=80,  exp=184, outchannels=80,  ks=3, SEModel=None,         NL="HS", s=1),
            bneck(inchannels=80,  exp=184, outchannels=80,  ks=3, SEModel=None,         NL="HS", s=1),
            bneck(inchannels=80,  exp=480, outchannels=112, ks=3, SEModel=SeModel(112), NL="HS", s=1),
            bneck(inchannels=112, exp=672, outchannels=112, ks=3, SEModel=SeModel(112), NL="HS", s=1),
            bneck(inchannels=112, exp=672, outchannels=160, ks=5, SEModel=SeModel(160), NL="HS", s=2),
            bneck(inchannels=160, exp=960, outchannels=160, ks=5, SEModel=SeModel(160), NL="HS", s=1),
            bneck(inchannels=160, exp=960, outchannels=160, ks=5, SEModel=SeModel(160), NL="HS", s=1)
        )
        
    
        self.conv2 = nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1),
            nn.BatchNorm2d(960),
            nn.Hardswish(inplace=True)
        )
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(960, 1280, 1),
            nn.Hardswish(inplace=True)
        )
        self.conv4 = nn.Conv2d(1280, 10, 1)


    def forward(self, x):
        out = self.conv1(x)
        out = self.bneck(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.shape[0], -1)
        return out



class MobileNetV3Small(nn.Module):
    def __init__(self, inchannel=1, classes=1000):
        super(MobileNetV3Small, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, 16, 3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.Hardswish()
        )

        self.bneck = nn.Sequential(
            bneck(inchannels=16, exp=16, outchannels=16, ks=3, SEModel=SeModel(16), NL="RE", s=1),
            bneck(inchannels=16, exp=72, outchannels=24, ks=3, SEModel=None,        NL="RE", s=1),
            bneck(inchannels=24, exp=88, outchannels=24, ks=3, SEModel=None,        NL="RE", s=1),
            bneck(inchannels=24, exp=96, outchannels=40, ks=5, SEModel=SeModel(40), NL="HS", s=1),
            bneck(inchannels=40, exp=240, outchannels=40, ks=5, SEModel=SeModel(40), NL="HS", s=1),
            bneck(inchannels=40, exp=240, outchannels=40, ks=5, SEModel=SeModel(40), NL="HS", s=1),
            bneck(inchannels=40, exp=120, outchannels=48, ks=5, SEModel=SeModel(48), NL="HS", s=1),
            bneck(inchannels=48, exp=144, outchannels=48, ks=5, SEModel=SeModel(48), NL="HS", s=1),
            bneck(inchannels=48, exp=288, outchannels=96, ks=5, SEModel=SeModel(96), NL="HS", s=2),
            bneck(inchannels=96, exp=576, outchannels=96, ks=5, SEModel=SeModel(96), NL="HS", s=1),
            bneck(inchannels=96, exp=576, outchannels=96, ks=5, SEModel=SeModel(96), NL="HS", s=1)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 576, kernel_size=1),
            nn.BatchNorm2d(576),
            nn.Hardswish(inplace=True)
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(576, 1280, kernel_size=1),
            nn.Hardswish(inplace=True)
        )
        
        self.conv4 = nn.Conv2d(1280, 10, kernel_size=1)
        
    
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.bneck(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = out.view(out.shape[0], -1)
        return out

def loadModel(model):
    if model == "MobileNetV3_l":
        model = MobileNetV3Large()
    elif model == "MobileNetV3_s":
        model = MobileNetV3Small()
    return model
