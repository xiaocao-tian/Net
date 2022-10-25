import argparse
import os
import time
import torch
import torch.nn as nn
from utils import selectDevice
from inceptionV1 import loadModel
from dataset import loadData
from val import val


def train(opt):
    model,aux, weights, data_dir, epochs, batch_size, learning_rete, resume, device, save_dir = opt.model, opt.aux, opt.weights, \
    opt.data_dir, opt.epochs, opt.batch_size, opt.learning_rate, opt.resume, opt.device, opt.save_dir

    # Directories
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    best, last = save_dir + "/best.pt", save_dir + "/last.pt"

    # Model
    model = loadModel(model)
    device = selectDevice(device)
    model = model.to(device)
    
    # Resume
    if resume:
        assert weights.endswith(".pt"), "No weight file found!"
        ckpt = torch.load(weights, map_location=device)
        model.load_state_dict(ckpt["model"])

    # Optimizer
    g0, g1, g2 = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v, nn.Parameter):
            g2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g0.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v, nn.Parameter):
            g1.append(v.weight)
    optimizer = torch.optim.Adam(g0, lr=learning_rete, betas=[0.9, 0.999])
    optimizer.add_param_group({"params": g1, "weight_decay": 0.0005})
    optimizer.add_param_group({"params": g2})
    del g0, g1, g2

    schedualer = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.3*epochs, gamma=0.1, last_epoch=-1)

    # Dataloader
    train_loader = loadData(data_dir, batch_size, "train-images.idx3-ubyte", "train-labels.idx1-ubyte", training=True)
    test_loader  = loadData(data_dir, batch_size, "t10k-images.idx3-ubyte",  "t10k-labels.idx1-ubyte",  training=False)

    # Optimizer
    start_epoch, best_acc = 0, 0.0
    if resume:
        if ckpt['optimizer']:
            optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_acc = ckpt['best_acc']
        del ckpt

    # Loss
    t0 = time.time()
    compute_loss = torch.nn.CrossEntropyLoss()

    # Start training
    print("Start training...")
    for epoch in range(start_epoch, epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device).float()/255.0
            labels = labels.to(device).long()
            pred = model(images)
            if aux:
                loss0 = compute_loss(pred[0], labels) 
                loss1 = compute_loss(pred[1], labels) * 0.3
                loss2 = compute_loss(pred[2], labels) * 0.3
                loss = loss0 + loss1 + loss2
            else:
                loss = compute_loss(pred, labels)

            loss.backward()
            optimizer.step()
            print("[Epoch: %d], Batch: %d/%d, Loss:%.2f, time:%2f" % (
                epoch, i + 1, len(train_loader), loss, time.time() - t0))
        schedualer.step()

        acc = val(model, test_loader, device)
        if acc > best_acc:
            best_acc = acc
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc
        }
        torch.save(ckpt, last)
        if best_acc == acc:
            torch.save(ckpt, best)
            print("TEST ACC: %f, Save Model!" %acc)
        else:
            print("TEST ACC: %f" % acc)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="InceptionV1", help='InceptionV1')
    parser.add_argument('--aux', nargs='?', const=True, default=False, help='aux training')
    parser.add_argument('--weights', type=str, default='./runs/weights/last.pt', help='initial weights path')
    parser.add_argument('--data_dir', type=str, default='./data', help='dataset.yaml path')
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=256, help='total batch size for all GPUs')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--resume', nargs='?', const=True, default=False, help='resume most recent training')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save_dir', default='runs/weights', help='save weights')
    opt = parser.parse_args()
    return opt

if __name__ == '__main__':
    opt = parse_opt()
    train(opt)
