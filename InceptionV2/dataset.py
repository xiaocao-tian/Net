import numpy as np
from torch.utils.data import DataLoader

def loadMNISTData(data_dir, image_file, label_file):
    with open(data_dir + '/' + label_file, 'rb') as lbpath:
        y_train = np.frombuffer(lbpath.read(), np.uint8, offset=8)
    with open(data_dir + '/' + image_file, 'rb') as lbpath:
        x_train = np.frombuffer(lbpath.read(), np.uint8, offset=16).reshape(len(y_train), 1, 28, 28)

    return x_train, y_train


class MNISTData:
    def __init__(self, data_dir, image_file, label_file):
        super(MNISTData, self).__init__()
        self.images, self.labels = loadMNISTData(data_dir, image_file, label_file)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, item):
        images = self.images[item]
        labels = self.labels[item]
        
        return images, labels


def loadData(data_dir, batch_size, image_file, label_file, training):
    dataset = MNISTData(data_dir, image_file, label_file)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=training)
    return loader
