import os.path
import time
from val import val
import torch.nn as nn
import torch
import argparse
from resnet import loadModel
from utils import selectDevice
from dataset import loadData


def train(opt):
	model, weights, data_dir, epochs, batch_size, learning_rate, resume, device, save_dir = opt.model, opt.weights, \
	opt.data_dir, opt.epochs, opt.batch_size, opt.learning_rate, opt.resume, opt.device, opt.save_dir
	
	if not os.path.exists(save_dir):
		os.makedirs(save_dir)
	best, last = save_dir + "/best.pt", save_dir + "/last.pt"
	model = loadModel(model)
	device = selectDevice(device)
	model = model.to(device)
	if resume:
		assert weights.endswith(".pt"), "No last.pt file"
		ckpt = torch.load(weights, map_location=device)		
		model.load_state_dict(ckpt["model"], strict=False)

	
	g0, g1, g2 = [], [], []
	for v in model.modules():
		if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
			g2.append(v.bias)
		if isinstance(v, nn.BatchNorm2d):
			g0.append(v.weight)
		elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
			g1.append(v.weight)

	optimizer = torch.optim.Adam(g0, lr=learning_rate, betas=(0.937, 0.999))
	optimizer.add_param_group({"params": g1, "weight_decay": 0.0005})
	optimizer.add_param_group({"params": g2})
	del g0, g1, g2
	
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs*0.3, gamma=0.1, last_epoch=-1)
	start_epoch, best_acc = 0, 0.0
	
	if resume:
		if ckpt["optimizer"]:
			optimizer.load_state_dict(ckpt['optimizer'])
			best_acc = ckpt['best_acc']
		start_epoch = ckpt["epoch"] + 1
		del ckpt

	train_loader = loadData(data_dir, batch_size, "train-images.idx3-ubyte", "train-labels.idx1-ubyte", training=True)
	test_loader  = loadData(data_dir, batch_size, "t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte",  training=False)

	t0 = time.time()
	compute_loss = torch.nn.CrossEntropyLoss()
	for epoch in range(start_epoch, epochs):
		model.train()
		for i, (images, labels) in enumerate(train_loader):
			optimizer.zero_grad()
			images = images.to(device, non_blocking=True).float()/255.0
			labels = labels.to(device, non_blocking=True).long()
			predictions = model(images)
			loss = compute_loss(predictions, labels)
			
			loss.backward()
			print("[Epoch: %d], Batch_size: %d/%d, Loss:%.2f, time:%2f" % (
				epoch, i + 1, len(train_loader), loss, time.time() - t0))
			
			optimizer.step()
		scheduler.step()
		
		acc = val(model, test_loader, device)
		if acc > best_acc:
			best_acc = acc
		ckpt = {
			'epoch': epoch,
			'best_acc': best_acc,
			'model': model.state_dict(),
			'optimizer': optimizer.state_dict()
		}
		torch.save(ckpt, last)
		if best_acc == acc:
			torch.save(ckpt, best)
			print("TEST: acc: %.2f, Save model" % acc)
		else:
			print("TEST: acc: %.2f" % acc)


def parse_opt():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', type=int, default=50, help='select 18, 34, 50, 101, 152')
	parser.add_argument('--weights', type=str, default='./runs/weights/last.pt', help='initial weights path')
	parser.add_argument('--data_dir', type=str, default='./data', help='dataset.yaml path')
	parser.add_argument('--epochs', type=int, default=300)
	parser.add_argument('--batch-size', type=int, default=256, help='total batch size for all GPUs')
	parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
	parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
	parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
	parser.add_argument('--save_dir', default='runs/weights', help='save weights')
	opt = parser.parse_args()
	return opt


if __name__ == '__main__':
	opt = parse_opt()
	train(opt)
