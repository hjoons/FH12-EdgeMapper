import torch
import h5py
import numpy as np
import random

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

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

def create_loader(input, labels, bs, workers=1):
    """
    Returns a DataLoader using CustomDataset. 
    Workers and batch size can also be specified.
    
    Args:
        input (list of torch.Tensor): A list of input PyTorch tensors.
        labels (list of torch.Tensor): A list of label PyTorch tensors.
        bs (integer): Specified batch size of DataLoader.
        workers (integer): Specified number of workers for DataLoader.
        
    Returns:
        torch.utils.data.DataLoader: DataLoader that can get batches of paired input and label tensors.
    """
    
    dataset = CustomDataset(input, labels)
    if workers == 1:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True)
    else:
        loader = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=workers)
    return loader
    

def get_tensors(mat_dir): 
    """
    Creates tensors given a path to the nyuv2 .mat file. 
    This method pulls from RGB images to create input tensors and
    pulls from the depth maps to create label tensors.
    This outputs an approximate 80-20 train-test-split.

    Args:
        mat_dir (string): Path to the specified nyuv2 .mat file.

    Returns:
        tuple: A tuple of lists of tensors.
            - list of torch.Tensor: Input training tensors.
            - list of torch.Tensor: Ground truth training tensors.
            - list of torch.Tensor: Input validation tensors.
            - list of torch.Tensor: Ground truth validation tensors.
    """
    mat_file = h5py.File('nyu_depth_v2_labeled.mat', 'r')
    rgb_images = mat_file['images'][:]
    depth_images = mat_file['depths'][:]
    
    X_train = rgb_images[0:1150].astype(np.uint8)
    X_test = rgb_images[1150::].astype(np.uint8)
    y_train = depth_images[0:1150].astype(np.uint8)
    y_test = depth_images[1150::].astype(np.uint8)
    
    transform = transforms.ToTensor()

    X_train_tensors = []
    for x in X_train:
        x = transform(x)
        x = x.transpose(0,1)
        x = x.transpose(1,2)
        X_train_tensors.append(x)

    X_test_tensors = []
    for x in X_test:
        x = transform(x)
        x = x.transpose(0,1)
        x = x.transpose(1,2)
        X_test_tensors.append(x)

    train_max = 0

    y_train_tensors = []
    for y in y_train:
        y = transform(y)
        train_max = torch.max(y) if torch.max(y) > train_max else train_max
        y_train_tensors.append(y / train_max)
        
    test_max = 0

    y_test_tensors = []
    for y in y_test:
        y = transform(y)
        test_max = torch.max(y) if torch.max(y) > test_max else test_max
        y_test_tensors.append(y / test_max)
    
    return X_train_tensors, y_train_tensors, X_test_tensors, y_test_tensors

def meanstd(tensors):
    """
    Calculates the mean and standard deviation per channel given a list of tensors.
    tensors is assumed to be an RGB image with 3 channels.

    Args:
        tensors (list of a list of torch.Tensor): A list of lists containing tensors. Each list is considered a channel.

    Returns:
        tuple: A tuple of mean and standard deviation.
            - tuple: The means of each respective channel.
                - float: The mean of the first channel (R).
                - float: The mean of the second channel (G).
                - float: The mean of the third channel (B).
            - tuple: The standard deviations of each respective channel.
                - float: The standard deviation of the first channel (R).
                - float: The standard deviation of the second channel (G).
                - float: The standard deviation of the third channel (B).
    """
    mean = [0, 0, 0]
    std = [0, 0, 0]
    for t in tensors:
        mean[0] += torch.sum(t[0, :, :]).item()
        mean[1] += torch.sum(t[1, :, :]).item()
        mean[2] += torch.sum(t[2, :, :]).item()
        
    mean = tuple(m / (len(tensors) * 640 * 480) for m in mean)
    
    
    for t in tensors:
        red_diff = (t[0, :, :] - mean[0]) ** 2    
        green_diff = (t[1, :, :] - mean[1]) ** 2
        blue_diff = (t[2, :, :] - mean[2]) ** 2
        
        std[0] += torch.sum(red_diff).item()
        std[1] += torch.sum(green_diff).item()
        std[2] += torch.sum(blue_diff).item()
    
    std = tuple(s / (len(tensors) * 640 * 480) for s in std)
    
    return mean, std

def normalize(images):
    """
    Given RGB images, returns normalized tensors using mean and standard deviation.

    Args:
        images (list of list of np.uint8): RGB images in the form of np.uint8 lists. 

    Returns:
        list of list of torch.Tensor: Returns a normalized list of list of tensors.
    """
    mean, std = meanstd(images)

    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ])

    tensors = []
    for x in images:
        x = np.transpose(x, (1, 2, 0))
        x = pipeline(x)
        tensors.append(x)
    
    return tensors