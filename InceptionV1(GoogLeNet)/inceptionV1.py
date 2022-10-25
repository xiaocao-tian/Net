import torch
import torch.nn as nn



class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3_redu, ch3x3, ch5x5_redu, ch5x5, pool_proj):
        super(Inception, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_redu, kernel_size=1),
            nn.BatchNorm2d(ch3x3_redu),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch3x3_redu, ch3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_redu, kernel_size=1),
            nn.BatchNorm2d(ch5x5_redu),
            nn.ReLU(inplace=True),

            nn.Conv2d(ch5x5_redu, ch5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], 1)


class InceptionAux(nn.Module):
    def __init__(self, in_channels, aux_dropout=0.7):
        super(InceptionAux, self).__init__()
        self.avg_pool = nn.AvgPool2d(kernel_size=5, stride=3)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 128, kernel_size=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.dropout = nn.Dropout(p=aux_dropout, inplace=True)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, 1000)

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.conv(out)
        out = out.view(out.shape[0], -1)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)

        return out


class InceptionV1(nn.Module):
    def __init__(self, aux_logits=False, dropout=0.4, classes=10):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),

            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        )

        self.inception3a = Inception(in_channels=192, ch1x1=64,  ch3x3_redu=96,  ch3x3=128, ch5x5_redu=16, ch5x5=32, pool_proj=32)
        self.inception3b = Inception(in_channels=256, ch1x1=128, ch3x3_redu=128, ch3x3=192, ch5x5_redu=32, ch5x5=96, pool_proj=64)
        self.max_pool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.inception4a = Inception(in_channels=480, ch1x1=192, ch3x3_redu=96, ch3x3=208, ch5x5_redu=16, ch5x5=48, pool_proj=64)
        self.inception4b = Inception(in_channels=512, ch1x1=160, ch3x3_redu=112, ch3x3=224, ch5x5_redu=24, ch5x5=64, pool_proj=64)
        self.inception4c = Inception(in_channels=512, ch1x1=128, ch3x3_redu=128, ch3x3=256, ch5x5_redu=24, ch5x5=64, pool_proj=64)
        self.inception4d = Inception(in_channels=512, ch1x1=112, ch3x3_redu=144, ch3x3=288, ch5x5_redu=32, ch5x5=64, pool_proj=64)
        self.inception4e = Inception(in_channels=528, ch1x1=256, ch3x3_redu=160, ch3x3=320, ch5x5_redu=32, ch5x5=128, pool_proj=128)
        self.max_pool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception(in_channels=832, ch1x1=256, ch3x3_redu=160, ch3x3=320, ch5x5_redu=32, ch5x5=128, pool_proj=128)
        self.inception5b = Inception(in_channels=832, ch1x1=384, ch3x3_redu=192, ch3x3=384, ch5x5_redu=48, ch5x5=128, pool_proj=128)
        self.avg_pool5 = nn.AvgPool2d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(1024, classes)

        if aux_logits:
            self.aux1 = InceptionAux(512)
            self.aux2 = InceptionAux(528)
        else:
            self.aux1 = None
            self.aux2 = None


    def forward(self, x):
        out = self.conv1(x)
        
        out = self.inception3a(out)
        out = self.inception3b(out)
        out = self.max_pool3(out)
        out = self.inception4a(out)
        aux0 = None
        if self.aux_logits:
            aux0 = self.aux1(out)
        out = self.inception4b(out)
        out = self.inception4c(out)
        out = self.inception4d(out)
        aux1 = None
        if self.aux_logits:
            aux1 = self.aux2(out)
        out = self.inception4e(out)
        out = self.max_pool4(out)
        out = self.inception5a(out)
        out = self.inception5b(out)
        out = self.avg_pool5(out)
        #out = self.dropout(out)
        out = out.view(out.shape[0], -1)

        out = self.fc(out)

        return [out, aux0, aux1] if self.aux_logits else out



def loadModel(model):
    if model == "InceptionV1":
        model = InceptionV1()
    return model
