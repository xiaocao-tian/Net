import numpy as np

def val(model, test_loader, device):
    model.eval()
    model.to(device)
    
    correct = 0.0
    sum_a = 0.0
    
    for i, (images, labels) in enumerate(test_loader):
        images = images.to(device, non_blocking=True).float() / 255.0
        predictions = model(images).detach()
        predictions = predictions.to("cpu")
        predictions = np.argmax(predictions, axis=-1)
        T = predictions == labels
        correct += np.sum(T.numpy(), axis=-1)
        sum_a += T.shape[0]
        
    return correct / sum_a