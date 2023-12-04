import torchsummary
import torch
from model import UNet

unet = UNet().to(torch.device("cuda"))
torchsummary.summary(unet, input_size=(3,640,480), batch_size=1)

