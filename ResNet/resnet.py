import torch.nn as nn


class BasicBlock(nn.Module):
	expansion = 1
	def __init__(self, inchannel, outchannel, stride=1):
		super(BasicBlock, self).__init__()

		self.inchannel = inchannel
		self.outchannel = outchannel
		self.stride = stride

		self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(outchannel)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(outchannel, outchannel*BasicBlock.expansion, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(outchannel*BasicBlock.expansion)


		if stride != 1 or inchannel != outchannel:
			self.shortcut = nn.Sequential(
				nn.Conv2d(inchannel, outchannel*BasicBlock.expansion, kernel_size=1, stride=stride),
				nn.BatchNorm2d(outchannel*BasicBlock.expansion)
			)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.stride != 1 or self.inchannel != self.outchannel:
			residual = self.shortcut(residual)

		out += residual
		out = self.relu(out)
		return out


class BottleNeck(nn.Module):
	expansion = 4
	def __init__(self, inchannel, outchannel, stride=1):
		super(BottleNeck, self).__init__()

		self.inchannel = inchannel
		self.outchannel = outchannel
		self.stride = stride

		self.conv1 = nn.Conv2d(inchannel, outchannel, kernel_size=1, bias=False)
		self.bn1 = nn.BatchNorm2d(outchannel)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(outchannel)
		self.conv3 = nn.Conv2d(outchannel, outchannel*BottleNeck.expansion, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(outchannel*BottleNeck.expansion)

		self.shortcut = nn.Sequential()

		if stride != 1 or inchannel != outchannel*BottleNeck.expansion:
			self.shortcut = nn.Sequential(
					nn.Conv2d(inchannel, outchannel*BottleNeck.expansion, kernel_size=1, stride=stride),
					nn.BatchNorm2d(outchannel*BottleNeck.expansion)
				)

	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)
		out = self.conv3(out)
		out = self.bn3(out)

		if self.stride != 1 or self.inchannel != self.outchannel*BottleNeck.expansion:
			residual = self.shortcut(residual)

		out += residual
		out = self.relu(out)
		return out

class ResNet(nn.Module):
	def __init__(self, block, layers, num_class=1000):
		super(ResNet, self).__init__()
		self.inchannels = 64
		self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
		self.bn = nn.BatchNorm2d(64)
		self.relu = nn.ReLU(inplace=True)
		self.max_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
		self.layer1 = self.make_layer(block, 64, layers[0])
		self.layer2 = self.make_layer(block, 128, layers[1], stride=1)
		self.layer3 = self.make_layer(block, 256, layers[2], stride=1)
		self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
		self.avg_pool = nn.AdaptiveAvgPool2d([1, 1])
		self.fc = nn.Linear(512 * block.expansion, 10)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()


	def forward(self, x):
		out = self.conv1(x)
		out = self.bn(out)
		out = self.relu(out)
		out = self.max_pool(out)
		out = self.layer1(out)
		out = self.layer2(out)
		out = self.layer3(out)
		out = self.layer4(out)
		out = self.avg_pool(out)
		out = out.view(x.size(0), -1)
		out = self.fc(out)

		return out

	def make_layer(self, block, channels, num_blocks, stride=1):
		strides = [stride] + [1] * (num_blocks-1)

		layers = []
		for stride in strides:
			layers.append(block(self.inchannels, channels, stride))
			self.inchannels = channels * block.expansion

		return nn.Sequential(*layers)



def loadModel(modelx):
	if modelx == 18:
		model = ResNet(BasicBlock, [2,2,2,2])
	elif modelx == 34:
		model = ResNet(BasicBlock, [3,4,6,3])
	elif modelx == 50:
		model = ResNet(BottleNeck, [3,4,6,3])
	elif modelx == 101:
		model = ResNet(BottleNeck, [3,4,23,3])
	elif modelx == 152:
		model = ResNet(BottleNeck, [3,8,36,3])
	return model































