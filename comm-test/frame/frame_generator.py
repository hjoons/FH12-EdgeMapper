# import cv as mkv_reader
import matplotlib.pyplot as plt
from matplotlib import image
import os
import cv2
from frame_helpers import colorize, convert_to_bgra_if_required
from pyk4a import PyK4APlayback
import h5py
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random
import torch
import argparse

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

def info(playback: PyK4APlayback):
    print(f"Record length: {playback.length / 1000000: 0.2f} sec")
          
# Load an MKV file
def create_frames(path: str):
    # Load an MKV file
    if os.path.exists(f"{path}"):
        mkv_file = f"{path}"
    else:
        print("That path doesn't exist bruh lol")
        exit()

    playback = PyK4APlayback(mkv_file)
    playback.open()
    playback.seek(5000000) # in microseconds, aka 5 seconds
    info(playback) # prints the recording length

    # print(type(playback))
    # Create a new HDF5 file
    file = h5py.File('output_1.h5', 'w')

    # Create datasets for RGB images and depth maps
    rgb_images = file.create_dataset('images', (0, 640, 480, 3), maxshape=(None, 640,480, 3))
    depth_images = file.create_dataset('depths', (0, 640, 480), maxshape=(None, 640, 480))

    i = 0
    while True:
        try:
            capture = playback.get_next_capture()

            if capture.color is not None:
                img_color = cv2.resize(cv2.cvtColor(convert_to_bgra_if_required(0, capture.color), cv2.COLOR_BGR2RGB)[80:720, 446: 926, 0:3], (480, 640))

                # Append the RGB image to the dataset
                rgb_images.resize(i + 1, axis=0)
                rgb_images[i] = img_color

            if capture.depth is not None:
                img_depth = cv2.resize(capture.transformed_depth[80:720, 446: 926], (480, 640))

                # Append the depth map to the dataset
                depth_images.resize(i + 1, axis=0)
                depth_images[i] = img_depth

            i += 1

            # key = cv2.waitKey(0)
            # if key != -1:
            #     break
        except EOFError:
            break

    # Close the HDF5 file
    file.close()

    # Close the MKV file
    playback.close()

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
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    return loader

def main(path: str):
    create_frames("C:/Users/vliew/Documents/UTAustin/Fall2023/SeniorDesign/output.mkv")
    x, y = create_data_set("output_1.h5")
    loader = create_data_loader(x, y)
    print("done")

if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--path", help="path to mkv file")
    args = argsparser.parse_args()
    main(args.path)