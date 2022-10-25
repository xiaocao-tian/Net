import torch.nn as nn
import torch


class BottleNeck(nn.Module):
    def __init__(self, inchannel, ch3x3_redu, ch3x3, ch3x3_redu_doub, ch3x3_doub, pool_proj):
        super(BottleNeck, self).__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(inchannel, ch3x3_redu, kernel_size=1),
            nn.BatchNorm2d(ch3x3_redu),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch3x3_redu, ch3x3, kernel_size=3, stride=2),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(inchannel, ch3x3_redu_doub, kernel_size=1),
            nn.BatchNorm2d(ch3x3_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch3x3_redu_doub, ch3x3_redu_doub, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch3x3_redu_doub, ch3x3_doub, kernel_size=3, stride=2),
            nn.BatchNorm2d(ch3x3_doub),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(inchannel, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        return torch.cat([out1, out2, out3], 1)


class InceptionAux(nn.Module):
    def __init__(self, inchannel, outchannel, classes=1000):
        super(InceptionAux, self).__init__()
        self.conv = nn.Sequential(
            nn.AdaptiveAvgPool2d([4, 4]),
            nn.Conv2d(inchannel, outchannel, kernel_size=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True)
        )
        
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 10)
        
    def forward(self, x):
        out = self.conv(x)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)
        
        return out



class InceptionA(nn.Module):
    def __init__(self, inchannel, ch1x1, ch3x3_redu, ch3x3, ch3x3_redu_doub, ch3x3_doub, pool_proj):
        super(InceptionA, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(inchannel, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(inchannel, ch3x3_redu, kernel_size=1),
            nn.BatchNorm2d(ch3x3_redu),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch3x3_redu, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(inchannel, ch3x3_redu_doub, kernel_size=1),
            nn.BatchNorm2d(ch3x3_redu_doub),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch3x3_redu_doub, ch3x3_redu_doub, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch3x3_redu_doub, ch3x3_doub, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3_doub),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(inchannel, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)

        return torch.cat([out1, out2, out3, out4], 1)


class InceptionB(nn.Module):
    def __init__(self, inchannel, ch1x1, ch7x7_redu, ch7x7, ch7x7_redu_doub, ch7x7_doub, pool_proj):
        super(InceptionB, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(inchannel, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(inchannel, ch7x7_redu, kernel_size=1),
            nn.BatchNorm2d(ch7x7_redu),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch7x7_redu, ch7x7_redu, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(ch7x7_redu),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch7x7_redu, ch7x7, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(ch7x7),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(inchannel, ch7x7_redu_doub, kernel_size=1),
            nn.BatchNorm2d(ch7x7_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch7x7_redu_doub, ch7x7_redu_doub, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(ch7x7_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch7x7_redu_doub, ch7x7_redu_doub, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(ch7x7_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch7x7_redu_doub, ch7x7_redu_doub, kernel_size=(1, 7), padding=(0, 3)),
            nn.BatchNorm2d(ch7x7_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch7x7_redu_doub, ch7x7_doub, kernel_size=(7, 1), padding=(3, 0)),
            nn.BatchNorm2d(ch7x7_doub),
            nn.ReLU(inplace=True)
        )
    
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(inchannel, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
       
    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        
        return torch.cat([out1, out2, out3, out4], 1)


class InceptionC(nn.Module):
    def __init__(self, inchannel, ch1x1, ch3x3_redu, ch3x3, ch3x3_redu_doub, ch3x3_doub, pool_proj):
        super(InceptionC, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(inchannel, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        
        self.branch2 = nn.Sequential(
            nn.Conv2d(inchannel, ch3x3_redu, kernel_size=1),
            nn.BatchNorm2d(ch3x3_redu),
            nn.ReLU(inplace=True)
        )
        
        self.branch2_1x3 = nn.Sequential(
            nn.Conv2d(ch3x3_redu, ch3x3//2, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(ch3x3//2),
            nn.ReLU(inplace=True)
        )
        
        self.branch2_3x1 = nn.Sequential(
            nn.Conv2d(ch3x3_redu, ch3x3//2, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(ch3x3//2),
            nn.ReLU(inplace=True)
        )
        
        self.branch3 = nn.Sequential(
            nn.Conv2d(inchannel, ch3x3_redu_doub, kernel_size=1),
            nn.BatchNorm2d(ch3x3_redu_doub),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(ch3x3_redu_doub, ch3x3_redu_doub, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3_redu_doub),
            nn.ReLU(inplace=True)
        )
        
        self.branch3_1x3 = nn.Sequential(
            nn.Conv2d(ch3x3_redu_doub, ch3x3_doub//2, kernel_size=(1,3), padding=(0,1)),
            nn.BatchNorm2d(ch3x3_doub//2),
            nn.ReLU(inplace=True)
        )
        
        self.branch3_3x1 = nn.Sequential(
            nn.Conv2d(ch3x3_redu_doub, ch3x3_doub//2, kernel_size=(3,1), padding=(1,0)),
            nn.BatchNorm2d(ch3x3_doub//2),
            nn.ReLU(inplace=True)
        )
        
        self.branch4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(inchannel, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)

        out2_1 = self.branch2(x)
        out2_2 = [self.branch2_1x3(out2_1), self.branch2_3x1(out2_1)]
        out2 = torch.cat(out2_2, 1)

        out3_1 = self.branch3(x)
        out3_2 = [self.branch3_1x3(out3_1), self.branch3_3x1(out3_1)]
        out3 = torch.cat(out3_2, 1)

        out4 = self.branch4(x)

        return torch.cat([out1, out2, out3, out4], 1)


class InceptionV3(nn.Module):
    def __init__(self, inchannel=3, classes=10):
        super(InceptionV3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
			
            nn.Conv2d(32, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
			
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
			
            nn.Conv2d(64, 80, kernel_size=3),
            nn.BatchNorm2d(80),
            nn.ReLU(inplace=True),
			
            nn.Conv2d(80, 192, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
			
            nn.Conv2d(192, 288, kernel_size=3, padding=1),
            nn.BatchNorm2d(288),
            nn.ReLU(inplace=True)
		)

        self.inception3a = InceptionA(inchannel=288,  ch1x1=64,  ch3x3_redu=64,  ch3x3=96,  ch3x3_redu_doub=64,  ch3x3_doub=96,  pool_proj=32)
        self.inception3b = InceptionA(inchannel=288,  ch1x1=64,  ch3x3_redu=64,  ch3x3=96,  ch3x3_redu_doub=64,  ch3x3_doub=96,  pool_proj=32)
        self.inception3c = InceptionA(inchannel=288,  ch1x1=64,  ch3x3_redu=64,  ch3x3=96,  ch3x3_redu_doub=64,  ch3x3_doub=96,  pool_proj=32)
        
        self.bottlenecka = BottleNeck(inchannel=288, ch3x3_redu=128, ch3x3=192, ch3x3_redu_doub=128, ch3x3_doub=192, pool_proj=384)
        
        self.inception5a = InceptionB(inchannel=768,  ch1x1=128, ch7x7_redu=192, ch7x7=256, ch7x7_redu_doub=192, ch7x7_doub=256, pool_proj=128)
        self.inception5b = InceptionB(inchannel=768,  ch1x1=128, ch7x7_redu=192, ch7x7=256, ch7x7_redu_doub=192, ch7x7_doub=256, pool_proj=128)
        self.inception5c = InceptionB(inchannel=768,  ch1x1=128, ch7x7_redu=192, ch7x7=256, ch7x7_redu_doub=192, ch7x7_doub=256, pool_proj=128)
        self.inception5d = InceptionB(inchannel=768,  ch1x1=128, ch7x7_redu=192, ch7x7=256, ch7x7_redu_doub=192, ch7x7_doub=256, pool_proj=128)
        self.inception5e = InceptionB(inchannel=768,  ch1x1=128, ch7x7_redu=192, ch7x7=256, ch7x7_redu_doub=192, ch7x7_doub=256, pool_proj=128)

        self.bottleneckb = BottleNeck(inchannel=768, ch3x3_redu=192, ch3x3=320, ch3x3_redu_doub=192, ch3x3_doub=320, pool_proj=640)

        self.inception2a = InceptionC(inchannel=1280, ch1x1=512, ch3x3_redu=320, ch3x3=512, ch3x3_redu_doub=320, ch3x3_doub=512, pool_proj=512)
        self.inception2b = InceptionC(inchannel=2048, ch1x1=512, ch3x3_redu=320, ch3x3=512, ch3x3_redu_doub=320, ch3x3_doub=512, pool_proj=512)

        self.avgpool = nn.AdaptiveAvgPool2d([1,1])
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, classes)
        
        self.aux = InceptionAux(768, 128)
        
        
    def forward(self, x):
        out = self.conv1(x)
        
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.inception3c(out)
        out = self.bottlenecka(out)

        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.inception5c(out)
        out = self.inception5d(out)
        out = self.inception5e(out)
        
        aux = self.aux(out)
        
        out = self.bottleneckb(out)
        
        out = self.inception2a(out)
        out = self.inception2b(out)

        out = self.avgpool(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.fc2(out)

        return [out, aux]
        
        

def loadModel(model):
	if model == "InceptionV3":
		model = InceptionV3()
	return model
