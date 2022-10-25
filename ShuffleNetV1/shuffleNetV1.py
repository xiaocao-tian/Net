import torch
import torch.nn as nn




class ShuffleChannel(nn.Module):
    def __init__(self, groups=3):
        super(ShuffleChannel, self).__init__()
        self.groups = groups
    
    def forward(self, x):
        bs, channels, width, height = x.shape
        channel_pre_group = int(channels/ self.groups)
        x = x.view(bs, self.groups, channel_pre_group,  width, height)
        x = x.transpose(1, 2).contiguous()
        x = x.view(bs, -1, width, height)
        
        return x


class ShuffleNetUnit(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1, stage=2, groups = 3):
        super(ShuffleNetUnit, self).__init__()
        
        self.gconv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannel//4, kernel_size=3, stride=1, padding=1, groups=groups),
            nn.BatchNorm2d(outchannel//4),
            nn.ReLU(inplace=True)
        )

        if stage == 2:
            self.gconv1 = nn.Sequential(
                nn.Conv2d(inchannel, outchannel//4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(outchannel//4),
                nn.ReLU(inplace=True)
            )

        self.shuffle_channel = ShuffleChannel(groups)
        
        self.dwconv = nn.Sequential(
            nn.Conv2d(outchannel//4, outchannel//4, kernel_size=3, stride=stride, padding=1, groups=groups),
            nn.BatchNorm2d(outchannel//4)
        )
        self.gconv2 = nn.Sequential(
            nn.Conv2d(outchannel//4, outchannel, kernel_size=1, stride=1, groups=groups),
            nn.BatchNorm2d(outchannel)
        )
        
        self.fusion = self._add
        self.shortcut = nn.Sequential()
        self.relu = nn.ReLU(inplace=True)
        
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            
            self.gconv2 = nn.Sequential(
                nn.Conv2d(outchannel//4, outchannel-inchannel, kernel_size=1, stride=1, groups=groups),
                nn.BatchNorm2d(outchannel-inchannel)
            )
        
            self.fusion = self._cat
        
    def _add(self, x, y):
        return torch.add(x, y)

    def _cat(self, x, y):
        return torch.cat([x, y], 1)
        
    def forward(self, x):
        shortout = self.shortcut(x)
        out = self.gconv1(x)
        out = self.shuffle_channel(out)
        out = self.dwconv(out)
        out = self.gconv2(out)
        out = self.fusion(out, shortout)
        out = self.relu(out)
        return out


class ShuffleNetV1(nn.Module):
    def __init__(self, inchannel, groups, repeats, classes=10):
        super(ShuffleNetV1, self).__init__()
        
        if groups == 1:
            outchannels = [24, 144, 288, 576]
        elif groups == 2:
            outchannels = [24, 200, 400, 800]
        elif groups == 3:
            outchannels = [24, 240, 480, 960]
        elif groups == 4:
            outchannels = [24, 272, 544, 1088]
        elif groups == 8:
            outchannels = [24, 384, 768, 1536]

        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannel, outchannels[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannels[0]),
            nn.ReLU(inplace=True)
        )
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.inchannel = outchannels[0]

        self.stage2 = self._make_layer(inchannel=self.inchannel, outchannel=outchannels[1], repeat=repeats[0], stage=2, groups=groups)
        self.stage3 = self._make_layer(inchannel=self.inchannel, outchannel=outchannels[2], repeat=repeats[1], stage=3, groups=groups)
        self.stage4 = self._make_layer(inchannel=self.inchannel, outchannel=outchannels[3], repeat=repeats[2], stage=4, groups=groups)
        
        self.global_pool = nn.AdaptiveAvgPool2d([1,1])
        self.fc = nn.Linear(outchannels[3], classes)
        
    def _make_layer(self, inchannel, outchannel, repeat, stage, groups):
        strides = [2] + [1] * repeat
        
        stages = []
        
        for stride in strides:
            stages.append(
                ShuffleNetUnit(
                    self.inchannel, 
                    outchannel, 
                    stride=stride, 
                    stage=stage, 
                    groups = groups
                    )
                )
            self.inchannel = outchannel
        
        return nn.Sequential(*stages)
        

    def forward(self, x):
        out = self.conv1(x)
        out = self.max_pool(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)
        out = self.global_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        
        return out


def loadModel(model):
    if model == "ShuffleNetV1":
        model = ShuffleNetV1(inchannel=1, groups=1, repeats=[3,7,3])
    return model
