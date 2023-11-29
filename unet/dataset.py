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

# Works with https://www.kaggle.com/datasets/soumikrakshit/nyu-depth-v2/data

_check_pil = lambda x: isinstance(x, Image.Image)

_check_np_img = lambda x: isinstance(x, np.ndarray)

class DepthDataset(Dataset):
    """
    Customized DataSet for grabbing images. DataLoader wraps around this Dataset class to generate batches.

    Args:
        Dataset (torch.utils.Dataset): torch.utils.data.Dataset superclass.
    """
    def __init__(self, data, nyu2_train, transform=None):
        """
        Initialize a DepthDataset using tuple generated by loadZipToMem().

        Args:
            data (dict): A dictionary mapping paths to image bytecode.
            nyu2_train (list of lists): Pairs of input and ground truth paths.
            transform (torchvision.transforms.Transform): Custom transformations that can be applied to images.
        """
        self.data, self.nyu_dataset = data, nyu2_train
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the entire Dataset.

        Returns:
            integer: Length of CustomDataset. 
        """
        return len(self.nyu_dataset)

    def __getitem__(self, idx):
        """
        Returns a tuple of the input tensors and ground truth tensors based on index.

        Args:
            idx (integer): The accessed index.

        Returns:
            dict: A containing "image" and "depth"
        """
        sample = self.nyu_dataset[idx]
        train_byte = BytesIO(self.data[sample[0]])
        depth_byte = BytesIO(self.data[sample[1]])
        
        image = Image.open(train_byte)
        depth = Image.open(depth_byte)
        sample = {"image": image, "depth": depth}
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
class ToTensor(object):
    def __init__(self, is_test=False):
        self.is_test = is_test

    def __call__(self, sample):
        image, depth = sample["image"], sample["depth"]

        image = self.to_tensor(image)

        if self.is_test:
            depth = self.to_tensor(depth).float() / 1000
        else:
            depth = self.to_tensor(depth).float() * 1000

        # put in expected range
        depth = torch.clamp(depth, 10, 1000)

        return {"image": image, "depth": depth}

    def to_tensor(self, pic):
        if not (_check_pil(pic) or _check_np_img(pic)):
            raise TypeError(
                "pic should be PIL Image or ndarray. Got {}".format(type(pic))
            )

        if isinstance(pic, np.ndarray):
            img = torch.from_numpy(pic.transpose((2, 0, 1)))

            return img.float().div(255)

        # handle PIL Image
        if pic.mode == "I":
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == "I;16":
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == "YCbCr":
            nchannel = 3
        elif pic.mode == "I;16":
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)

        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(255)
        else:
            return img

def loadZipToMem(zip_file,train_test="train"):
    """
    Takes in a zip file path and creates a list of paired paths mapping
    an input image to the ground truth. The data dictionary maps the
    names of each file to a bytecode representation of the image.

    Args:
        zip_file (string): Path to zip file.
        train_test (str, optional): Specifies if obtaining testing or training data. Defaults to "train".

    Returns:
        tuple: Returns a data dictionary and a list of lists.
        - dict: A dictionary mapping paths to image bytecode.
        - list of lists: A list of lists containing pairs of input and ground truth paths.
    """
    # Load zip file into memory
    
    print("Loading dataset zip file...", end="")

    input_zip = ZipFile(zip_file)
    data = {name: input_zip.read(name) for name in input_zip.namelist()}
    nyu2_train = list(
        (
            row.split(",")
            for row in (data[f"data/nyu2_{train_test}.csv"]).decode("utf-8").split("\n")
            if len(row) > 0
        )
    )

    nyu2_train = shuffle(nyu2_train, random_state=0)

    print("Loaded ({0}).".format(len(nyu2_train)))

    return data, nyu2_train

class RandomHorizontalFlip(object):
    def __call__(self, sample):

        img, depth = sample["image"], sample["depth"]

        if not _check_pil(img):
            raise TypeError("Expected PIL type. Got {}".format(type(img)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            depth = depth.transpose(Image.FLIP_LEFT_RIGHT)

        return {"image": img, "depth": depth}


class RandomChannelSwap(object):
    def __init__(self, probability):

        self.probability = probability
        self.indices = list(permutations(range(3), 3))

    def __call__(self, sample):

        image, depth = sample["image"], sample["depth"]

        if not _check_pil(image):
            raise TypeError("Expected PIL type. Got {}".format(type(image)))
        if not _check_pil(depth):
            raise TypeError("Expected PIL type. Got {}".format(type(depth)))

        if random.random() < self.probability:
            image = np.asarray(image)
            image = Image.fromarray(
                image[..., list(self.indices[random.randint(0, len(self.indices) - 1)])]
            )

        return {"image": image, "depth": depth}

def getNoTransform(is_test=False):
    return transforms.Compose([ToTensor(is_test=is_test)])

def getDefaultTrainTransform():
    return transforms.Compose(
        [RandomHorizontalFlip(), RandomChannelSwap(0.5), ToTensor()]
    )

def createTrainLoader(path, samples=2000, batch_size=1, workers=1):
    """
    Returns a training and validation DataLoader using
    DepthDataset. The size of the dataset, batch size,
    and workers can all be specified.

    Args:
        path (string): Path to the zip file with all data.
        samples (int, optional): Number of samples encompassing test and validation.
        batch_size (int, optional): Batch size of data and validation. Defaults to 1.
        workers (int, optional): Workers for data and validation. Defaults to 1.
    Returns:
        tuple: Tuple of DataLoaders.
        - torch.utils.data.DataLoader: DataLoader that can get batches of paired input and label tensors.
        - torch.utils.data.DataLoader: DataLoader that can get batches of paired input and label tensors.
    """
    
    random.seed(42)
    
    data, nyu2_train = loadZipToMem(path)
    transformed_training = DepthDataset(
        data, nyu2_train, transform=getDefaultTrainTransform()
    )
    transformed_testing = DepthDataset(
        data, nyu2_train, transform=getNoTransform()
    )
    
    random_indices = set()
    while len(random_indices) < samples:
        random_indices.add(random.randint(0, len(transformed_training) - 1))

    random_indices = list(random_indices) 
    
    training = torch.utils.data.Subset(transformed_training, random_indices)
    testing = torch.utils.data.Subset(transformed_testing, random_indices)
    
    total = len(training)
    train_len = int(0.8*total)
    
    train_set, _ = torch.utils.data.random_split(training, [train_len, total-train_len])
    _, val_set = torch.utils.data.random_split(testing, [train_len, total-train_len])
    
    print(f"Training set size: {len(train_set)}  Validation set size: {len(val_set)}")
    
    return DataLoader(train_set, batch_size, shuffle=True), DataLoader(val_set, batch_size, shuffle=False)

def createTestLoader(path, samples=500, batch_size=1, workers=1):
    """
    Returns the testing DataLoader. The batch size and
    worksers can be specified.

    Args:
        path (string): Path to the zip file with all data.
        batch_size (int, optional): Batch size. Defaults to 1.
        workers (int, optional): Workers. Defaults to 1.
    Returns:
        torch.utils.data.DataLoader: DataLoader that can get batches of paired input and label tensors.
    """
    
    random.seed(42)

    data, nyu2_train = loadZipToMem(path)
    transformed_testing = DepthDataset(
        data, nyu2_train, transform=getNoTransform()
    )
    
    random_indices = set()
    while len(random_indices) < samples:
        random_indices.add(random.randint(0, len(transformed_testing) - 1))

    random_indices = list(random_indices) 
    
    testing = torch.utils.data.Subset(transformed_testing, random_indices) 
    
    return DataLoader(testing, batch_size=batch_size, shuffle=False)

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
    
def createH5TrainLoader(path=None, samples=2000, batch_size=1, workers=1):
    transformed_training = H5DepthDataset(path, transform=getDefaultTrainTransform())
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    
    return DataLoader(transformed_training, batch_size=batch_size, shuffle=True), DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)

def createH5TestLoader(path=None, samples=500, batch_size=1, workers=1):
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    return DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)