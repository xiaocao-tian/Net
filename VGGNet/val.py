import numpy as np


def val(model, test_loader, device):
    model.eval()
    
    correct = 0.0
    sum_all = 0.0
    for (images, labels) in test_loader:
        images = images.to(device, non_blocking=True).float()/255.0
        pred = model(images).detach()
        pred = pred.to('cpu')
        pred = np.argmax(pred, axis=-1) # 获取最大值对应的索引
        T = pred == labels
        correct += np.sum(T.numpy(), axis=-1)
        sum_all += T.shape[0]
    return correct / sum_all
        
        
