import torch.nn as nn




class BottleNeck(nn.Module):
    def __init__(self, t, c1, c2, s):
        super(BottleNeck, self).__init__()
        
        self.stride = s
        self.inchannels = c1
        self.outchannels = c2
        
        self.block = nn.Sequential(
            nn.Conv2d(c1, c1*t, kernel_size=1),
            nn.BatchNorm2d(c1*t),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(c1*t, c1*t, kernel_size=3, stride=s, padding=1, groups=c1*t),
            nn.BatchNorm2d(c1*t),
            nn.ReLU6(inplace=True),
            
            nn.Conv2d(c1*t, c2, kernel_size=1),
            nn.BatchNorm2d(c2),
        )


    def forward(self, x):
        residual = self.block(x)
        
        if self.stride == 1 and self.inchannels == self.outchannels:
            residual += x
            
        return residual


class MobileNetV2(nn.Module):
    def __init__(self, inchannels, classes=10):
        super(MobileNetV2, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(inchannels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )
        
        self.layer1 = self._make_layer(t=1, c1=32,  c2=16,  n=1, s=1)
        self.layer2 = self._make_layer(t=6, c1=16,  c2=24,  n=2, s=1)
        self.layer3 = self._make_layer(t=6, c1=24,  c2=32,  n=3, s=1)
        self.layer4 = self._make_layer(t=6, c1=32,  c2=64,  n=4, s=1)
        self.layer5 = self._make_layer(t=6, c1=64,  c2=96,  n=3, s=1)
        self.layer6 = self._make_layer(t=6, c1=96,  c2=160, n=3, s=1)
        self.layer7 = self._make_layer(t=6, c1=160, c2=320, n=1, s=2)
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )
        
        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.conv3 = nn.Conv2d(1280, classes, 1)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.conv2(out)
        out = self.avg_pool(out)
        out = self.conv3(out)
        out = out.view(out.shape[0], -1)
        return out
        
        
    def _make_layer(self, t, c1, c2, n, s):
        layers = []
        layers.append(BottleNeck(t, c1, c2, s))
        
        for i in range(1, n):
            layers.append(BottleNeck(t, c2, c2, s=1))

        return nn.Sequential(*layers)

def loadModel(model):
    if model == "mobileNetV2":
        model = MobileNetV2(inchannels=1)
    return model






