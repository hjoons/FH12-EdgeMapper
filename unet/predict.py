from model import UNet
import torch
import torch.nn.functional as F
import time
import os
import pandas as pd
import dataset
import matplotlib.pyplot as plt

class CustomDataset(Dataset):
    """
    Customized DataSet for grabbing images. DataLoader wraps around this Dataset class to generate batches.

    Args:
        Dataset (torch.utils.Dataset): torch.utils.data.Dataset superclass.
    """
    def __init__(self, data, labels):
        """
        Initialize a CustomDataset using input tensors and label tensors.

        Args:
            data (list of list of torch.Tensor): Input tensors.
            labels (list of torch.Tensor): Label tensors.
        """
        self.data = data
        self.labels = labels
        self.transform = transforms.RandomHorizontalFlip()

    def __len__(self):
        """
        Returns the length of the entire Dataset.

        Returns:
            integer: Length of CustomDataset. 
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a tuple of the input tensors and ground truth tensors based on index.

        Args:
            idx (integer): The accessed index.

        Returns:
            tuple: A tuple of the input and ground truth tensors.
                - torch.Tensor: Input tensor.
                - torch.Tensor: Ground truth tensor. 
        """
        
        img = self.data[idx]
        truth = self.labels[idx]
        
        if self.transform:
            # to_pil = transforms.ToPILImage()
            # to_tensor = transforms.ToTensor()
            
            # img = to_pil(img)
            # img = self.transform(img)
            # img = to_tensor(img)
            p = random.random()
            if p >.5:
                img = torch.flip(img, dims=[2])
                truth = torch.flip(truth, dims=[2])
            
        return img, truth

def create_data_set(file_name):
    mat_file = h5py.File(f'{file_name}', 'r')
    rgb_images = mat_file['images'][:]
    depth_images = mat_file['depths'][:]

    transform = transforms.ToTensor()
    X_train = rgb_images
    y_train = depth_images

    X_train_tensors = []
    for x in X_train:
        x = transform(x)
        X_train_tensors.append(x)
    y_train_tensors = []
    train_max = 0
    for y in y_train:
        y = transform(y)
        train_max = torch.max(y) if torch.max(y) > train_max else train_max
        y_train_tensors.append(y / train_max)
    mat_file.close()

    return X_train_tensors, y_train_tensors

def create_data_loader(x, y):
    dataset = CustomDataset(x, y)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Now using device: {device}")

# create model, to be replaced with model checkpoint
model = UNet().to(torch.device(device))
model.eval()
print('Model on GPU in eval mode')

# load dataset
pth = '/home/orin/Documents/FH12_23-24/FH12-EdgeMapper/comm-test/frame/output_1.h5'
x, y = create_data_set(pth)
loader = create_data_loader(x, y)
print("Dataloader created")

# create plt
fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

# forward pass through model, show CPU time, GPU time, FPS of predicted depth output
with torch.no_grad():
        for batch_idx, (image, truth) in enumerate(loader):
            start_time = time.perf_counter()
            image = image.to(torch.device(device))
            truth = truth.to(torch.device(device))
            cpu_time = time.perf_counter()
            
            outputs = model(image)

            total_time = time.perf_counter - start_time
            print(f"Model ran for batch {batch_idx}")
            print(f'FPS: {1/total_time}')

