import torch
import time
import os
import pandas as pd
import dataset
import matplotlib.pyplot as plt

from model import UNet
from torch.utils.tensorboard import SummaryWriter
from losses import ssim as ssim_criterion
from losses import depth_loss as gradient_criterion
from dataset import getTrainingTestingData
from utils import DepthNorm

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
    
    # X_train, y_train, X_test, y_test = dataset.get_tensors(os.path.expanduser("~/Projects/data"))

    # training = dataset.create_loader(X_train, y_train, bs=4)
    # testing = dataset.create_loader(X_test, y_test, bs=4)

    # Load data
    print("Loading Data ...")
    trainloader, testloader = getTrainingTestingData(os.path.expanduser("~/Projects/data/NYU_DepthV2/data/nyu_depth.zip"), batch_size=4)
    print("Dataloaders ready ...")
    num_trainloader = len(trainloader)
    num_testloader = len(testloader)
    
    model = UNet().to(torch.device(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    # Loss functions
    l1_criterion = torch.nn.L1Loss()
    # criterion = custom_loss

    # out = model(X_test[0].to('cuda').unsqueeze(0))
    # print(out[0].shape,y_test[0].shape)
    # l = criterion(out[0].cpu(),y_test[0])
    
    
    train_loss = []
    test_loss = []
    batch_iter = 0
    for epoch in range(epochs):
        # train_iter = iter(trainloader)
        # total_batches = len(trainloader)
        time_start = time.perf_counter()
        model = model.train()
        running_loss = 0
        for batch_idx, batch in enumerate(trainloader):
            optimizer.zero_grad()

            image_x = torch.Tensor(batch["image"]).to(device)
            depth_y = torch.Tensor(batch["depth"]).to(device=device)

            normalized_depth_y = DepthNorm(depth_y)

            preds = model(image_x)

            # calculating the losses
            l1_loss = l1_criterion(preds, normalized_depth_y)

            ssim_loss = torch.clamp(
                (1 - ssim_criterion(preds, normalized_depth_y, 1000.0 / 10.0)) * 0.5,
                min=0,
                max=1,
            )

            gradient_loss = gradient_criterion(normalized_depth_y, preds, device=device)

            net_loss = (
                (1.0 * ssim_loss)
                + (1.0 * torch.mean(gradient_loss))
                + (0.1 * torch.mean(l1_loss))
            )
            cpu_loss = net_loss.cpu().detach().numpy()
            writer.add_scalar("Loss/batch_train",cpu_loss,batch_iter)
            running_loss += cpu_loss

            net_loss.backward()

            optimizer.step()
            batch_iter += 1
        time_end = time.perf_counter()
        train_loss.append(running_loss / len(trainloader))
        print(f'epoch: {epoch + 1} train loss: {running_loss / len(trainloader)} time: {time_end - time_start}')

        model.eval()
        with torch.no_grad():
            running_test_loss = 0
            for batch_idx, batch in enumerate(testloader):
                image_x = torch.Tensor(batch["image"]).to(device)
                depth_y = torch.Tensor(batch["depth"]).to(device=device)

                normalized_depth_y = DepthNorm(depth_y)

                preds = model(image_x)

                # calculating the losses
                l1_loss = l1_criterion(preds, normalized_depth_y)

                ssim_loss = torch.clamp(
                    (1 - ssim_criterion(preds, normalized_depth_y, 1000.0 / 10.0)) * 0.5,
                    min=0,
                    max=1,
                )

                gradient_loss = gradient_criterion(normalized_depth_y, preds, device=device)

                net_loss = (
                    (1.0 * ssim_loss)
                    + (1.0 * torch.mean(gradient_loss))
                    + (0.1 * torch.mean(l1_loss))
                )
                cpu_loss = net_loss.cpu().detach().numpy()
                running_test_loss += cpu_loss
            test_loss.append(running_test_loss / len(testloader))
            print(f'testing loss: {running_test_loss / len(testloader)}')

            batch = next(iter(testloader))
            image_x = torch.Tensor(batch["image"]).to(device)
            depth_y = torch.Tensor(batch["depth"]).to(device=device)
            for i in range(3):
                out = model(image_x[i].unsqueeze(0))
                normalized_depth_y = DepthNorm(depth_y[i])

                preds = model(image_x[i].unsqueeze(0))

                # calculating the losses
                l1_loss = l1_criterion(preds, normalized_depth_y)

                ssim_loss = torch.clamp(
                    (1 - ssim_criterion(preds, normalized_depth_y, 1000.0 / 10.0)) * 0.5,
                    min=0,
                    max=1,
                )

                gradient_loss = gradient_criterion(normalized_depth_y.unsqueeze(0), preds, device=device)

                net_loss = (
                    (1.0 * ssim_loss)
                    + (1.0 * torch.mean(gradient_loss))
                    + (0.1 * torch.mean(l1_loss))
                )
                cpu_loss = net_loss.cpu().detach().numpy()
                fig,ax = plt.subplots(1,3,figsize=(16,8))
                ax[0].imshow(image_x[i].permute(1,2,0).cpu())
                ax[1].imshow(depth_y[i][0].cpu())
                ax[2].imshow(out.cpu()[0][0])
                ax[2].set_title(f"loss:{net_loss}")
                fig.savefig(f'test_img{i}_epoch_{epoch}.png')
        
        writer.add_scalar('Loss/train', (running_loss / len(trainloader)), global_step=epoch)
        writer.add_scalar('Loss/validation', (running_test_loss / len(testloader)), global_step=epoch)
        
        if (epoch + 1) % 5 == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }
            
            torch.save(checkpoint, f'epoch_{epoch + 1}.pt')
        
            csv_train = pd.DataFrame({"1e-3": train_loss})
            csv_test = pd.DataFrame({"1e-3": test_loss})
            csv_train.to_csv(f"logs/training_loss.csv")
            csv_test.to_csv(f"logs/testing_loss.csv")
        
    writer.close()