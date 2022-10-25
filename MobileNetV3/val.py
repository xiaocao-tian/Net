import numpy as np



def val(model, loader, device):
    model.eval()
    
    correct = 0.0
    sum_all = 0.0
    for images, labels in loader:
        images = images.to(device).float()/255.0
        pred = model(images).detach()
        pred = pred.to('cpu')
        pred = np.argmax(pred, axis=-1)
        T = pred == labels
        correct += np.sum(T.numpy(), axis=-1)
        sum_all += T.shape[0]
    return correct/sum_all
    