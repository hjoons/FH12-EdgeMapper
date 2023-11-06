import train
import utils
import torch
import dataset
from model import UNet

if __name__ == '__main__':
    train.train(lr=1e-3, epochs=150)
    load_net = UNet()

    checkpoint = torch.load('checkpoints/epoch_50.pt')
    load_net.load_state_dict(checkpoint['model_state_dict'])
    
    X_train, y_train, X_test, y_test = dataset.get_tensors('./nyu_depth_v2_labeled.mat')
    testing = dataset.create_loader(X_test, y_test, bs=1)

    utils.evaluate(load_net, testing)