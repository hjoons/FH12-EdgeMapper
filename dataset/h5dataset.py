import h5py
from PIL import Image
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from dataset.dataset import getDefaultTrainTransform, getNoTransform
from torch.utils.data import DataLoader, Dataset


class H5DepthDataset(Dataset):
    """
    Class that represents a depth dataset from a .h5 file.

    Attributes:
        h5_path (str): path to the .h5 file
        transform (torchvision.transforms): transforms to be applied to the dataset

    Methods:
        __len__(): returns the length of the dataset
        __getitem__(index): returns the sample at the index
    """
    def __init__(self, h5_path, transform=None):
        """
        Constructor for the H5DepthDataset class

        Args:
            h5_path (str): path to the .h5 file
            transform (torchvision.transforms): transforms to be applied to the dataset

        Returns:
            None
        """
        h5_file = h5py.File(h5_path, 'r')
        self.h5_file = h5_file
        self.transform = transform
        
    def __len__(self):
        """
        Returns the length of the dataset

        Args:
            None
        
        Returns:
            int: length of the dataset
        """ 
        return len(self.h5_file['images'])
    
    def __getitem__(self, index):
        """
        Returns the sample at the index

        Args:
            index (int): index of the sample

        Returns:
            dict: sample at the index
        """
        img = self.h5_file['images'][index]
        gt = self.h5_file['depths'][index]

        pil_img = Image.fromarray(img, 'RGB')
        pil_gt = Image.fromarray(gt, 'L')

        sample = {"image": pil_img, "depth": pil_gt}
    
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
def createH5TrainLoader(path, batch_size=1):
    """
    Function that creates a data loader from a .h5 file given a file path

    Args:
        path (str): path to the .h5 file
        batch_size (int): batch size

    Returns:
        DataLoader: data loader
    """
    transformed_training = H5DepthDataset(path, transform=getDefaultTrainTransform())
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    
    return DataLoader(transformed_training, batch_size=batch_size, shuffle=True), DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)

def createH5TestLoader(path, batch_size=1):
    """
    Function that creates a data loader from a .h5 file given a file path

    Args:
        path (str): path to the .h5 file
        batch_size (int): batch size

    Returns:
        DataLoader: data loader
    """
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    return DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)
