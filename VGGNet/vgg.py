import torch.nn as nn


# 为满足MNIST数据集，最后一个池化删掉
cfg = {
		'A': [64, 	  'M', 128, 	 'M', 256, 256, 	 	  'M', 512, 512, 	 	   'M', 512, 512, 			],#'M'],
		'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 	 	  'M', 512, 512, 	 	   'M', 512, 512, 			],#'M'],
		'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 	  'M', 512, 512, 512, 	   'M', 512, 512, 512, 		],#'M'],
		'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, ],#'M'],
}


class VGGNet(nn.Module):
	def __init__(self, cfg):
		super(VGGNet, self).__init__()
		self.layer = self._make_layers(cfg)
		self.fc1 = nn.Linear(512, 4096)
		
		self.dropout = nn.Dropout()
		self.fc2 = nn.Linear(4096, 4096)
		self.fc3 = nn.Linear(4096, 10)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()



	def forward(self, x):
		out = self.layer(x)
		out = out.view(out.shape[0], -1)
		out = self.fc1(out)
		out = self.dropout(out)
		out = self.fc2(out)
		out = self.dropout(out)
		out = self.fc3(out)
		
		return out

	def _make_layers(self, cfg):
		layers = []
		inchannel = 1
		for l in cfg:
			if l == "M":
				layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
			else:
				layers.append(nn.Conv2d(inchannel, l, kernel_size=3, padding=1))
				layers.append(nn.BatchNorm2d(l))
				layers.append(nn.ReLU(inplace=True))
				inchannel = l
		return nn.Sequential(*layers)


def loadModel(model):

	if model == 11:
		model = VGGNet(cfg['A'])
	elif model == 13:
		model = VGGNet(cfg['B'])
	elif model == 16:
		model = VGGNet(cfg['D'])
	elif model == 19:
		model = VGGNet(cfg['E'])
		
	return model
