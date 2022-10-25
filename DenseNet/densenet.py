import torch
import torch.nn as nn


dense_dict = {
    121: [6, 12, 24, 16],
    169: [6, 12, 32, 32],
    201: [6, 12, 48, 32],
    264: [6, 12, 64, 48],
}

class BottleNeck(nn.Module):
    def __init__(self, in_channel, growth_rate):
        super(BottleNeck, self).__init__()
        inner_channel = growth_rate * 4
        self.bottle_neck = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, inner_channel, kernel_size=1),

            nn.BatchNorm2d(inner_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channel, growth_rate, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return torch.cat([x, self.bottle_neck(x)], 1)


class transition_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(transition_layer, self).__init__()
        self.trainsition = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1),
            nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        return self.trainsition(x)


class DensetNet(nn.Module):
    def __init__(self, block, growth_rate=32, nblocks=[6, 12, 24, 16], reduction=0.5, classes=10):
        super(DensetNet, self).__init__()

        self.growth_rate = growth_rate

        in_channel = growth_rate * 2

        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),
            nn.Conv2d(1, in_channel, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.feature = nn.Sequential()

        for index in range(len(nblocks) - 1):
            self.feature.add_module("Dense_block_layer_%d" % index, self._make_layer(block, in_channel, nblocks[index]))

            in_channel += self.growth_rate * nblocks[index]

            out_channel = int(in_channel * reduction)
            self.feature.add_module("transition_layer_%d" % index, transition_layer(in_channel, out_channel))
            in_channel = out_channel

        self.feature.add_module("Dense_block_layer_%d" % (len(nblocks)-1), self._make_layer(block, in_channel, nblocks[len(nblocks)-1]))
        in_channel += growth_rate * (nblocks[len(nblocks) - 1])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.softmax = nn.Linear(in_channel, classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.feature(out)
        out = self.avg_pool(out)
        out = out.view(out.size()[0], -1)
        out = self.softmax(out)
        return out

    def _make_layer(self, block, in_channel, nblock):
        desnet_block = nn.Sequential()
        for index in range(nblock):
            desnet_block.add_module("bottle_neck_layer_%d" % index, block(in_channel, self.growth_rate))
            in_channel += self.growth_rate

        return desnet_block


def loadModel(model):
    model = DensetNet(BottleNeck, growth_rate=32, nblocks=dense_dict[model])

    return model
