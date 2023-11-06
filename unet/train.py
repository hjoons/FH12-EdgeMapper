import torch
import time
import os
import pandas as pd
import dataset

from model import UNet
from torch.utils.tensorboard import SummaryWriter

def custom_loss(output, target):
    di = target - output
    n = 640*480
    di2 = torch.pow(di, 2)
    first_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = .5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n**2)
    loss = first_term - second_term
    return loss.mean()

def train(lr=1e-3, epochs=200):
    writer = SummaryWriter('logs')
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Now using device: {device}")
    
    X_train, y_train, X_test, y_test = dataset.get_tensors('./nyu_depth_v2_labeled.mat')
    training = dataset.create_loader(X_train, y_train, bs=4)
    testing = dataset.create_loader(X_test, y_test, bs=4)
    
    model = UNet().to(torch.device(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = custom_loss
    
    train_loss = []
    test_loss = []
    for epoch in range(epochs):
        time_start = time.perf_counter()
        model = model.train()
        running_loss = 0
        for batch_idx, (image, truth) in enumerate(training):
            image = image.to(torch.device(device))
            truth = truth.to(torch.device(device))

            optimizer.zero_grad()

            outputs = model(image)
            
            loss = criterion(outputs, truth)
            cpu_loss = loss.cpu().detach().numpy()
            
            running_loss += cpu_loss

            loss.backward()

            optimizer.step()
        time_end = time.perf_counter()
        train_loss.append(running_loss / len(training))
        print(f'epoch: {epoch + 1} train loss: {running_loss / len(training)} time: {time_end - time_start}')

        model.eval()
        with torch.no_grad():
            running_test_loss = 0
            for batch_idx, (image, truth) in enumerate(testing):
                image = image.to(torch.device(device))
                truth = truth.to(torch.device(device))
                
                outputs = model(image)
                
                loss = criterion(outputs, truth)
                cpu_loss = loss.cpu().detach().numpy()
                running_test_loss += cpu_loss
            test_loss.append(running_test_loss / len(testing))
            print(f'testing loss: {running_test_loss / len(testing)}')
        
        writer.add_scalar('Loss/train', (running_loss / len(training)), global_step=epoch)
        writer.add_scalar('Loss/validation', (running_test_loss / len(testing)), global_step=epoch)
        
        if (epoch + 1) % 50 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            
            torch.save(checkpoint, os.path.join(os.getcwd(), 'checkpoints', f'epoch_{epoch + 1}.pt'))
        
            csv_train = pd.DataFrame({"1e-3": train_loss})
            csv_test = pd.DataFrame({"1e-3": test_loss})
            csv_train.to_csv(f"logs/training_loss.csv")
            csv_test.to_csv(f"logs/testing_loss.csv")
        
    writer.close()