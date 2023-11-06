import train
import utils
import torch
import dataset
from model import UNet

if __name__ == '__main__':
    # train.train(lr=1e-3, epochs=50)
    load_net = UNet()

    checkpoint = torch.load('checkpoints/epoch_50.pt')
    load_net.load_state_dict(checkpoint['model_state_dict'])
    
    X_train, y_train, X_test, y_test = dataset.get_tensors('./nyu_depth_v2_labeled.mat')
    testing = dataset.create_loader(X_test, y_test, bs=1)

    load_net.eval()
    d1 = 0
    d2 = 0
    d3 = 0
    for batchidx, batch in enumerate(testing):
        print(f"{batchidx + 1} / {len(testing)}")
        inputs, targets = batch
        with torch.no_grad():
            outputs = load_net(inputs)

            d1 += utils.threshold_percentage(outputs, targets, 1.25)
            d2 += utils.threshold_percentage(outputs, targets, 1.5625)
            d3 += utils.threshold_percentage(outputs, targets, 1.953125)
    
    d1 = d1 / len(testing)
    d2 = d2 / len(testing)
    d3 = d3 / len(testing)
    deltas = (d1, d2, d3)
    print(deltas)

        
            
    
    