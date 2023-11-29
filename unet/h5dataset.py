import torch
import numpy as np
import random
import h5py

from zipfile import ZipFile
from sklearn.utils import shuffle
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from io import BytesIO
from itertools import permutations

class H5DepthDataset(Dataset):
    
    def __init__(self, h5_path, transform=None):
        h5_file = h5py.File(h5_path, 'r')
        self.h5_file = h5_file
        self.transform = transform
        
    def __len__(self):
        return len(self.h5_file['images'])
    
    def __getitem__(self, index):
        sample = {"image": self.h5_file['images'][index], "depth": self.h5_file['depths'][index]}
    
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
def createTrainLoader(path, samples=2000, batch_size=1, workers=1):
    transformed_training = H5DepthDataset(path, transform=getDefaultTrainTransform())
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    
    return DataLoader(transformed_training, batch_size=batch_size, shuffle=True), DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)

def createTestLoader(path, samples=500, batch_size=1, workers=1):
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    return DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)