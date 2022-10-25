import torch
import torch.nn as nn


def channelSplit(x, split):
    
    assert x.size(1)  == split*2
    
    return torch.split(x, split, dim=1)


def channelShuffle(x, groups):
    bs, channel, width, height = x.shape
    channel_per_group = channel // groups
    x = x.view(bs, groups, channel_per_group, width, height)
    x = x.transpose(1, 2).contiguous()
    x = x.view(bs, -1, width, height)
    
    return x
 

class ShuffleNetUnit(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ShuffleNetUnit, self).__init__()
        
        self.inchannel = inchannel
        self.stride = stride

        if stride == 1:
        
            inchannel = inchannel//2
            
            self.residual = nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=stride, padding=1, groups=inchannel),
                nn.BatchNorm2d(inchannel),
                nn.Conv2d(inchannel, outchannel//2, kernel_size=1),
                nn.BatchNorm2d(outchannel//2),
                nn.ReLU(inplace=True),
            )
            
            self.shortcut = nn.Sequential()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=1),
                nn.BatchNorm2d(inchannel),
                nn.ReLU(inplace=True),
                nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=stride, padding=1, groups=inchannel),
                nn.BatchNorm2d(inchannel),
                nn.Conv2d(inchannel, outchannel//2, kernel_size=1),
                nn.BatchNorm2d(outchannel//2),
                nn.ReLU(inplace=True),
            )
            
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, inchannel, kernel_size=3, stride=stride, padding=1, groups=inchannel),
                nn.BatchNorm2d(inchannel),
                nn.Conv2d(inchannel, outchannel//2, kernel_size=1),
                nn.BatchNorm2d(outchannel//2),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        if self.stride == 1:
            shortout, residual = channelSplit(x, self.inchannel//2)
        else:
            shortout = residual = x
        
        shortout = self.shortcut(shortout)
        residual = self.residual(residual)
        out = torch.cat([shortout, residual], 1)
        
        out = channelShuffle(out, 2)
        
        return out
        
        

class shuffleNetV2(nn.Module):
    def __init__(self, inchannel=3, groups=1, classes=10):
        super(shuffleNetV2, self).__init__()
        
        if groups == 0.5:
            outchannel = [24, 48, 96, 192, 1024]
        elif groups == 1:
            outchannel = [24, 116, 232, 464, 1024]
        elif groups == 1.5:
            outchannel = [24, 176, 352, 704, 1024]
        elif groups == 2:
            outchannel = [24, 224, 488, 976, 2048]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )
        
        self.inchannel = outchannel[0]
        
        self.stage2 = self._make_layers(outchannel[1], repeat=3)
        self.stage3 = self._make_layers(outchannel[2], repeat=7)
        self.stage4 = self._make_layers(outchannel[3], repeat=3)
        
        self.conv5 = nn.Sequential(
            nn.Conv2d(outchannel[3], outchannel[4], kernel_size=1),
            nn.BatchNorm2d(outchannel[4]),
            nn.ReLU(inplace=True),
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(outchannel[4], classes)
        
    
    def _make_layers(self, outchannel, repeat):
        stages = []
        stages.append(ShuffleNetUnit(self.inchannel, outchannel, stride=2))
        self.inchannel = outchannel
        
        while repeat:
            stages.append(ShuffleNetUnit(outchannel, outchannel, stride=1))
            repeat -= 1
        
        return nn.Sequential(*stages)

    def forward(self, x):
        out = self.conv1(x)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.conv5(out)
        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        
        return out


def loadModel(model):
    if model == "shuffleNetV2":
        model = shuffleNetV2(inchannel=1, groups=0.5)
    return model
    
    
    
