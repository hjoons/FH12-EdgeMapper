import torch
import torch.nn.functional as F
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import h5dataset

from utils import DepthNorm
from losses import ssim, depth_loss
from model import UNet
from torch.utils.tensorboard import SummaryWriter

# def train(lr=1e-3, epochs=75, samples=2500, batch_size=4):
#     writer = SummaryWriter('logs')
    
#     device = (
#         "cuda"
#         if torch.cuda.is_available()
#         else "mps"
#         if torch.backends.mps.is_available()
#         else "cpu"
#     )
#     print(f"Now using device: {device}")
    
#     print("Loading data ...")
#     train_loader, val_loader = dataset.createTrainLoader("./nyu_data.zip", samples=samples, batch_size=batch_size)
    
#     test_loader = dataset.createTestLoader("./nyu_data.zip", batch_size=batch_size)
#     print("Test loader len: ", len(test_loader))
    
#     print("DataLoaders now ready ...")
#     num_trainloader = len(train_loader)
#     num_testloader = len(test_loader)
        
#     model = UNet().to(torch.device(device))
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
#     l1_criterion = torch.nn.L1Loss()
    
#     # batch_iter = 0
#     train_loss = []
#     test_loss = []
#     for epoch in range(epochs):
#         time_start = time.perf_counter()
#         model = model.train()
#         running_loss = 0
#         for batch_idx, batch in enumerate(train_loader):
#             optimizer.zero_grad()

#             image = torch.Tensor(batch["image"]).to(device)
#             depth = torch.Tensor(batch["depth"]).to(device)
            
#             normalized_depth = DepthNorm(depth)
            
#             pred = model(image)
            
#             l1_loss = l1_criterion(pred, normalized_depth)
            
#             ssim_loss = torch.clamp(
#                 (1 - ssim(pred, normalized_depth, 1000.0 / 10.0)) * 0.5,
#                 min=0,
#                 max=1,
#             )
            
#             gradient_loss = depth_loss(normalized_depth, pred, device=device)
            
#             net_loss = (
#                 (1.0 * ssim_loss)
#                 + (1.0 * torch.mean(gradient_loss))
#                 + (0.1 * torch.mean(l1_loss))
#             )
            
#             cpu_loss = net_loss.cpu().detach().numpy()
#             # writer.add_scalar("Loss/batch_train",cpu_loss,batch_iter)
#             running_loss += cpu_loss

#             net_loss.backward()

#             optimizer.step()
#             # batch_iter += 1

#         train_loss.append(running_loss / num_trainloader)
#         print(f'epoch: {epoch + 1} train loss: {running_loss / num_trainloader}')

#         model.eval()
#         with torch.no_grad():
#             running_test_loss = 0
#             for batch_idx, batch in enumerate(test_loader):
#                 image = torch.Tensor(batch["image"]).to(device)
#                 depth = torch.Tensor(batch["depth"]).to(device=device)

#                 normalized_depth = DepthNorm(depth)

#                 pred = model(image)

#                 # calculating the losses
#                 l1_loss = l1_criterion(pred, normalized_depth)

#                 ssim_loss = torch.clamp(
#                     (1 - ssim(pred, normalized_depth, 1000.0 / 10.0)) * 0.5,
#                     min=0,
#                     max=1,
#                 )

#                 gradient_loss = depth_loss(normalized_depth, pred, device=device)

#                 net_loss = (
#                     (1.0 * ssim_loss)
#                     + (1.0 * torch.mean(gradient_loss))
#                     + (0.1 * torch.mean(l1_loss))
#                 )
                
#                 cpu_loss = net_loss.cpu().detach().numpy()
#                 running_test_loss += cpu_loss
#             test_loss.append(running_test_loss / num_testloader)
#             time_end = time.perf_counter()
#             print(f'testing loss: {running_test_loss / num_testloader} time: {time_end - time_start}')
            

#             # batch = next(iter(test_loader))
#             # image_x = torch.Tensor(batch["image"]).to(device)
#             # depth_y = torch.Tensor(batch["depth"]).to(device=device)
#             # for i in range(3):
#             #     out = model(image_x[i].unsqueeze(0))
#             #     normalized_depth = DepthNorm(depth_y[i])

#             #     pred = model(image_x[i].unsqueeze(0))

#             #     # calculating the losses
#             #     l1_loss = l1_criterion(pred, normalized_depth)

#             #     ssim_loss = torch.clamp(
#             #         (1 - ssim(pred, normalized_depth, 1000.0 / 10.0)) * 0.5,
#             #         min=0,
#             #         max=1,
#             #     )

#             #     gradient_loss = depth_loss(normalized_depth.unsqueeze(0), pred, device=device)

#             #     net_loss = (
#             #         (1.0 * ssim_loss)
#             #         + (1.0 * torch.mean(gradient_loss))
#             #         + (0.1 * torch.mean(l1_loss))
#             #     )
#             #     cpu_loss = net_loss.cpu().detach().numpy()
#             #     fig,ax = plt.subplots(1,3,figsize=(16,8))
#             #     ax[0].imshow(image_x[i].permute(1,2,0).cpu())
#             #     ax[1].imshow(depth_y[i][0].cpu())
#             #     ax[2].imshow(out.cpu()[0][0])
#             #     ax[2].set_title(f"loss:{net_loss}")
#             #     fig.savefig(f'test_img{i}_epoch_{epoch}.png')
        
#         writer.add_scalar('Loss/train', (running_loss / num_trainloader), global_step=epoch)
#         writer.add_scalar('Loss/validation', (running_test_loss / num_testloader), global_step=epoch)
        
#         if (epoch + 1) % 50 == 0:
#             checkpoint = {
#                 'epoch': epoch + 1,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict()
#             }
            
#             torch.save(checkpoint, os.path.join(os.getcwd(), 'checkpoints', f'epoch_{epoch + 1}.pt'))
        
#             csv_train = pd.DataFrame({"1e-3": train_loss})
#             csv_test = pd.DataFrame({"1e-3": test_loss})
#             csv_train.to_csv(f"logs/training_loss.csv")
#             csv_test.to_csv(f"logs/testing_loss.csv")
        
#     writer.close()

def train_with_local(file_path, model_path, lr=1e-3, epochs=1, batch_size=1):
    # writer = SummaryWriter('logs')
    
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Now using device: {device}")
    
    print("Loading data ...")
    train_loader, val_loader = h5dataset.createH5TrainLoader(path=file_path, batch_size=1)

    # custom_loader = dataset.createCustomDataLoader(f'{file_path}')
    print("Custom loader len: ", len(train_loader))
    
    print("DataLoaders now ready ...")
    num_trainloader = len(train_loader)
    model = UNet().to(torch.device(device))
    model.load_state_dict(torch.load(f'{model_path}')['model_state_dict'])
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    l1_criterion = torch.nn.L1Loss()
    
    # batch_iter = 0
    train_loss = []
    test_loss = []
    print(f"About to train")
    for epoch in range(epochs):
        time_start = time.perf_counter()
        model = model.train()
        running_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            image = torch.Tensor(batch['image']).to(device)
            depth = torch.Tensor(batch['depth']).to(device)

            # print(f"putting images on device")
            #image = image.to(device)
            #depth = truth.to(device)

            normalized_depth = DepthNorm(depth)
            
            pred = model(image)

            l1_loss = l1_criterion(pred, normalized_depth)
            
            ssim_loss = torch.clamp(
                (1 - ssim(pred, normalized_depth, 1000.0 / 10.0)) * 0.5,
                min=0,
                max=1,
            )
            
            gradient_loss = depth_loss(normalized_depth, pred, device=device)
            
            net_loss = (
                (1.0 * ssim_loss)
                + (1.0 * torch.mean(gradient_loss))
                + (0.1 * torch.mean(l1_loss))
            )
            
            cpu_loss = net_loss.cpu().detach().numpy()
            # writer.add_scalar("Loss/batch_train",cpu_loss,batch_iter)
            running_loss += cpu_loss

            net_loss.backward()

            optimizer.step()
            # batch_iter += 1

        train_loss.append(running_loss / num_trainloader)
        print(f'epoch: {epoch + 1} train loss: {running_loss / num_trainloader}')

        # model.eval()
        # with torch.no_grad():
        #     running_test_loss = 0
        #     for batch_idx, (image, depth) in enumerate(custom_loader):
        #         image = torch.Tensor(image).to(device)
        #         depth = torch.Tensor(depth).to(device)

        #         normalized_depth = DepthNorm(depth)

        #         pred = model(image)

        #         # calculating the losses
        #         l1_loss = l1_criterion(pred, normalized_depth)

        #         ssim_loss = torch.clamp(
        #             (1 - ssim(pred, normalized_depth, 1000.0 / 10.0)) * 0.5,
        #             min=0,
        #             max=1,
        #         )

        #         gradient_loss = depth_loss(normalized_depth, pred, device=device)

        #         net_loss = (
        #             (1.0 * ssim_loss)
        #             + (1.0 * torch.mean(gradient_loss))
        #             + (0.1 * torch.mean(l1_loss))
        #         )
                
        #         cpu_loss = net_loss.cpu().detach().numpy()
        #         running_test_loss += cpu_loss
        #     test_loss.append(running_test_loss / num_trainloader)
        #     time_end = time.perf_counter()
        #     print(f'testing loss: {running_test_loss / num_trainloader} time: {time_end - time_start}')
        
        # # writer.add_scalar('Loss/train', (running_loss / num_trainloader), global_step=epoch)
        # # writer.add_scalar('Loss/validation', (running_test_loss / num_trainloader), global_step=epoch)
        
        # if (epoch + 1) % 50 == 0:
        
        #     csv_train = pd.DataFrame({"1e-3": train_loss})
        #     csv_test = pd.DataFrame({"1e-3": test_loss})
        #     csv_train.to_csv(f"logs_custom/training_loss.csv")
        #     csv_test.to_csv(f"logs_custom/testing_loss.csv")
        
    # writer.close()
