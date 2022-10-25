import torch
import torch.nn as nn


class BasicConv2d(nn.Module):
    def __init__(self, inchannel, outchannel, ks=3, stride=1, padding=0):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=ks, stride=stride, padding=padding),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class FactConv7x7(nn.Module):
    def __init__(self, inchannel, ch1x7, ch7x1):
        super(FactConv7x7, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, ch1x7, kernel_size =(1,7), padding=(0, 3)),
            nn.BatchNorm2d(ch1x7),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch1x7, ch7x1, kernel_size =(7,1), padding=(3, 0)),
            nn.BatchNorm2d(ch7x1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class FactConv1x3(nn.Module):
    def __init__(self, inchannel=256, outchannel=256):
        super(FactConv1x3, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(1,3), padding=(0, 1)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)

class FactConv3x1(nn.Module):
    def __init__(self, inchannel=256, outchannel=256):
        super(FactConv3x1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=(3,1), padding=(1, 0)),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return self.conv(x)


class Stem_V4_V2(nn.Module):
    def __init__(self, inchannel):
        super(Stem_V4_V2, self).__init__()
        self.conv1 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=32, ks=3, stride=2, padding=1),
            BasicConv2d(inchannel=32, outchannel=32, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=32, outchannel=64, ks=3, stride=1, padding=1)
        )
        
        self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv2 = BasicConv2d(inchannel=64, outchannel=96, ks=3, stride=1, padding=1)
        
        self.branch1 = nn.Sequential(
            BasicConv2d(inchannel=160, outchannel=64, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=64, outchannel=96, ks=3, stride=1, padding=1)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=160, outchannel=64, ks=1, stride=1, padding=0),
            FactConv7x7(inchannel=64, ch1x7=64, ch7x1=64),
            BasicConv2d(inchannel=64, outchannel=96, ks=3, stride=1, padding=1)
        )
        
        self.conv3 = BasicConv2d(inchannel=192, outchannel=192, ks=3, stride=1, padding=1)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):
        out = self.conv1(x)
        out = torch.cat([self.max_pool2(out), self.conv2(out)], 1)
        out = torch.cat([self.branch1(out), self.branch2(out)], 1)
        out = torch.cat([self.max_pool3(out), self.conv3(out)], 1)
        
        return out


class InceptionA_V4(nn.Module):
    def __init__(self, inchannel):
        super(InceptionA_V4, self).__init__()
        
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=96, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=64, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=64,  outchannel=96, ks=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=64, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=64,  outchannel=96, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=96,  outchannel=96, ks=3, stride=1, padding=1)
        )
        
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inchannel=inchannel,  outchannel=96, ks=1, stride=1, padding=0)
        )
    
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        return torch.cat([out1, out2, out3, out4],1)
        

class ReductionA_V4_V1_V2(nn.Module):
    def __init__(self, inchannel, k=192, l=224, m=256, n=384):
        super(ReductionA_V4_V1_V2, self).__init__()
        
        self.branch1 = BasicConv2d(inchannel=inchannel,  outchannel=n, ks=3, stride=1, padding=1)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel,  outchannel=k, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=k,  outchannel=l, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=l,  outchannel=m, ks=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        return torch.cat([out1, out2, out3], 1)


class InceptionB_V4(nn.Module):
    def __init__(self, inchannel):
        super(InceptionB_V4, self).__init__()
        
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=384, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0),
            FactConv7x7(inchannel=192, ch1x7=224, ch7x1=256),
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0),
            FactConv7x7(inchannel=192, ch1x7=192, ch7x1=224),
            FactConv7x7(inchannel=224, ch1x7=224, ch7x1=256),
        )
        
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inchannel=inchannel, outchannel=128, ks=1, stride=1, padding=0),
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        return torch.cat([out1, out2, out3, out4], 1)


class ReductionB_V4(nn.Module):
    def __init__(self, inchannel):
        super(ReductionB_V4, self).__init__()
        
        self.branch1 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=192,       outchannel=192, ks=3, stride=2, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
            FactConv7x7(inchannel=256, ch1x7=256, ch7x1=320),
            BasicConv2d(inchannel=320, outchannel=320, ks=3, stride=2, padding=1)
        )
        
        self.branch3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        
        return torch.cat([out1, out2, out3], 1)


class InceptionC_V4(nn.Module):
    def __init__(self, inchannel):
        super(InceptionC_V4, self).__init__()
        
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0)
        
        self.branch2 = BasicConv2d(inchannel=inchannel, outchannel=384, ks=1, stride=1, padding=0)
        self.branch2_1x3 = FactConv1x3(inchannel=384, outchannel=256)
        self.branch2_3x1 = FactConv3x1(inchannel=384, outchannel=256)
        
        self.branch3 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=384, ks=1, stride=1, padding=0),
            FactConv1x3(inchannel=384, outchannel=448),
            FactConv3x1(inchannel=448, outchannel=512)
        )
        self.branch3_1x3 = FactConv1x3(inchannel=512, outchannel=256)
        self.branch3_3x1 = FactConv3x1(inchannel=512, outchannel=256)

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        
        out2 = self.branch2(x)
        out2_1 = self.branch2_1x3(out2)
        out2_2 = self.branch2_3x1(out2)
        
        out3 = self.branch3(x)
        out3_1 = self.branch3_1x3(out3)
        out3_2 = self.branch3_3x1(out3)
        
        out4 = self.branch4(x)
        
        return torch.cat([out1, out2_1, out2_2, out3_1, out3_2, out4], 1)

class InceptionV4(nn.Module):
    def __init__(self, inchannel, classes=10):
        super(InceptionV4, self).__init__()
        self.stem = Stem_V4_V2(inchannel)
        self.inceptionA = nn.Sequential(
            InceptionA_V4(384),
            InceptionA_V4(384),
            InceptionA_V4(384),
            InceptionA_V4(384)
        )

        self.reductionA = ReductionA_V4_V1_V2(inchannel=384, k=192, l=224, m=256, n=384)

        self.inceptionB = nn.Sequential(
            InceptionB_V4(1024),
            InceptionB_V4(1024),
            InceptionB_V4(1024),
            InceptionB_V4(1024),
            InceptionB_V4(1024),
            InceptionB_V4(1024),
            InceptionB_V4(1024)
        )

        self.reductionB = ReductionB_V4(inchannel=1024)
        
        self.inceptionC = nn.Sequential(
            InceptionC_V4(1536),
            InceptionC_V4(1536),
            InceptionC_V4(1536)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])
        
        self.dropout = nn.Dropout(p=0.8)
        
        self.fc = nn.Linear(1536, classes)
        
    def forward(self, x):
        out = self.stem(x)
        out = self.inceptionA(out)
        out = self.reductionA(out)

        out = self.inceptionB(out)
        out = self.reductionB(out)

        out = self.inceptionC(out)

        out = self.avg_pool(out)
        out = out.view(out.shape[0], -1)
        out = self.dropout(out)
        out = self.fc(out)

        return out


class Stem_V1(nn.Module):
    def __init__(self, inchannel, classes=10):
        super(Stem_V1, self).__init__()
        self.conv = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=32, ks=3, stride=2, padding=1),
            BasicConv2d(inchannel=32,        outchannel=32, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=32,        outchannel=64, ks=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(inchannel=64,        outchannel=80,  ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=80,        outchannel=192, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=192,       outchannel=256, ks=3, stride=1, padding=1),
        )
    
    def forward(self, x):
        return self.conv(x)


class InceptionResNetA_V1(nn.Module):
    def __init__(self, inchannel):
        super(InceptionResNetA_V1, self).__init__()
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=32, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=32, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=32, outchannel=32, ks=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=32, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=32, outchannel=32, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=32, outchannel=32, ks=3, stride=1, padding=1)
        )

        self.conv = BasicConv2d(inchannel=96, outchannel=256, ks=1, stride=1, padding=0)
    
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], 1)
        out = self.conv(out)
        
        return out + x


class InceptionResNetB_V1(nn.Module):
    def __init__(self, inchannel):
        super(InceptionResNetB_V1, self).__init__()
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=128, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=128, ks=1, stride=1, padding=0),
            FactConv7x7(inchannel=128, ch1x7=128, ch7x1=128)
        )
        
        self.conv = BasicConv2d(inchannel=256, outchannel=896, ks=1, stride=1, padding=0)

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], 1)
        out = self.conv(out) * 0.1
        
        return out + x


class ReductionB_V1(nn.Module):
    def __init__(self, inchannel):
        super(ReductionB_V1, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=256,       outchannel=384, ks=3, stride=2, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=256,       outchannel=256, ks=3, stride=2, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=256,       outchannel=256, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=256,       outchannel=256, ks=3, stride=2, padding=1)
        )

        self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        return torch.cat([out1, out2, out3, out4], 1)
        

class InceptionResNetC_V1(nn.Module):
    def __init__(self, inchannel):
        super(InceptionResNetC_V1, self).__init__()
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0),
            FactConv1x3(inchannel=192, outchannel=192),
            FactConv3x1(inchannel=192, outchannel=192)
        )
        
        self.conv = BasicConv2d(inchannel=384, outchannel=1792, ks=1, stride=1, padding=0)
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], 1)
        out = self.conv(out)
        
        return out + x


class InceptionResNetV1(nn.Module):
    def __init__(self, inchannel, classes=10):
        super(InceptionResNetV1, self).__init__()
        self.stem = Stem_V1(inchannel=inchannel)
        self.inceptionresnetA = nn.Sequential(
            InceptionResNetA_V1(256),
            InceptionResNetA_V1(256),
            InceptionResNetA_V1(256),
            InceptionResNetA_V1(256),
            InceptionResNetA_V1(256)
        )
        
        self.reductionresnetA = ReductionA_V4_V1_V2(inchannel=256, k=192, l=192, m=256, n=384)
        
        self.inceptionresnetB = nn.Sequential(
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896),
            InceptionResNetB_V1(896)
        )
        
        self.reductionresnetB = ReductionB_V1(inchannel=896)
        
        self.inceptionresnetC = nn.Sequential(
            InceptionResNetC_V1(1792),
            InceptionResNetC_V1(1792),
            InceptionResNetC_V1(1792),
            InceptionResNetC_V1(1792),
            InceptionResNetC_V1(1792)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(1792, classes)
        
    def forward(self, x):
        out = self.stem(x)
        
        out = self.inceptionresnetA(out)
        out = self.reductionresnetA(out)
        
        out = self.inceptionresnetB(out)
        out = self.reductionresnetB(out)

        out = self.inceptionresnetC(out)
        
        out = self.avg_pool(out)

        out = out.view(out.shape[0], -1)
        out = self.fc(out)
        
        return out



class InceptionResNetA_V2(nn.Module):
    def __init__(self, inchannel):
        super(InceptionResNetA_V2, self).__init__()
        
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=32, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=32, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=32, outchannel=32, ks=3, stride=1, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=32, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=32, outchannel=48, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=48, outchannel=64, ks=3, stride=1, padding=1)
        )
        
        self.conv = BasicConv2d(inchannel=128, outchannel=384, ks=1, stride=1, padding=0)
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out = torch.cat([out1, out2, out3], 1)
        out = self.conv(out)
        
        return out + x
        

class InceptionResNetB_V2(nn.Module):
    def __init__(self, inchannel):
        super(InceptionResNetB_V2, self).__init__()
        
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=128, ks=1, stride=1, padding=0),
            FactConv7x7(inchannel=128, ch1x7=160, ch7x1=192)
        )
        
        self.conv = BasicConv2d(inchannel=384, outchannel=1152, ks=1, stride=1, padding=0)
    
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], 1)
        out = self.conv(out) * 0.1
        
        return out + x
        
   
class ReductionB_V2(nn.Module):
    def __init__(self, inchannel):
        super(ReductionB_V2, self).__init__()
        self.branch1 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=256,       outchannel=384, ks=3, stride=2, padding=1)
        )
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=256,       outchannel=288, ks=3, stride=2, padding=1)
        )
        
        self.branch3 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=256, ks=1, stride=1, padding=0),
            BasicConv2d(inchannel=256,       outchannel=288, ks=3, stride=1, padding=1),
            BasicConv2d(inchannel=288,       outchannel=320, ks=3, stride=2, padding=1)
        )
        
        self.branch4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        return torch.cat([out1, out2, out3, out4], 1)
        

class InceptionResNetC_V2(nn.Module):
    def __init__(self, inchannel):
        super(InceptionResNetC_V2, self).__init__()
        self.branch1 = BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0)
        
        self.branch2 = nn.Sequential(
            BasicConv2d(inchannel=inchannel, outchannel=192, ks=1, stride=1, padding=0),
            FactConv1x3(inchannel=192, outchannel=224),
            FactConv3x1(inchannel=224, outchannel=256)
        )
        
        self.conv = BasicConv2d(inchannel=448, outchannel=2144, ks=1, stride=1, padding=0)
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out = torch.cat([out1, out2], 1)
        out = self.conv(out) * 0.1
        
        return out + x



class InceptionResNetV2(nn.Module):
    def __init__(self, inchannel, classes=10):
        super(InceptionResNetV2, self).__init__()
        self.stem = Stem_V4_V2(inchannel)
        self.inceptionresnetA = nn.Sequential(
            InceptionResNetA_V2(384),
            InceptionResNetA_V2(384),
            InceptionResNetA_V2(384),
            InceptionResNetA_V2(384),
            InceptionResNetA_V2(384)
        )
        
        self.reductionresnetA = ReductionA_V4_V1_V2(inchannel=384, k=256, l=256, m=384, n=384)
        
        self.inceptionresnetB = nn.Sequential(
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152),
            InceptionResNetB_V2(1152)
            
        )
        
        self.reductionresnetB = ReductionB_V2(inchannel=1152)
        
        self.inceptionresnetC = nn.Sequential(
            InceptionResNetC_V2(2144),
            InceptionResNetC_V2(2144),
            InceptionResNetC_V2(2144),
            InceptionResNetC_V2(2144),
            InceptionResNetC_V2(2144)
        )
        
        self.avg_pool = nn.AdaptiveAvgPool2d([1,1])
        self.dropout = nn.Dropout(p=0.8)
        self.fc = nn.Linear(2144, classes)
    
    def forward(self, x):
        out = self.stem(x)

        out = self.inceptionresnetA(out)
        out = self.reductionresnetA(out)
        
        out = self.inceptionresnetB(out)
        out = self.reductionresnetB(out)
        
        out = self.inceptionresnetC(out)

        out = self.avg_pool(out)

        out = out.view(out.shape[0], -1)
        
        out = self.dropout(out)
        out = self.fc(out)

        
        return out
        

def loadModel(model):
    if model == "InceptionV4":
        model = InceptionV4(inchannel=1)
    elif model == "InceptionResNetV1":
        model = InceptionResNetV1(inchannel=1)
    elif model == "InceptionResNetV2":
        model = InceptionResNetV2(inchannel=1)
        
    return model
