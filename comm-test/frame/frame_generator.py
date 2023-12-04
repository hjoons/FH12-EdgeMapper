import matplotlib.pyplot as plt
from matplotlib import image
import os
import cv2
# from frame_helpers import colorize, convert_to_bgra_if_required

import pyk4a
from pyk4a import PyK4A, Config, FPS, DepthMode, ColorResolution

import h5py
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import random

# import torch
# import argparse

# from unet.model import UNet

import torch
from torch import nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, up_sample_mode):
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self, out_classes=1, up_sample_mode='conv_transpose'):
        super(UNet, self).__init__()
        self.up_sample_mode = up_sample_mode
        # Downsampling Path
        self.down_conv1 = DownBlock(3, 64)
        self.down_conv2 = DownBlock(64, 128)
        self.down_conv3 = DownBlock(128, 256)
        self.down_conv4 = DownBlock(256, 512)
        # Bottleneck
        self.double_conv = DoubleConv(512, 1024)
        # Upsampling Path
        self.up_conv4 = UpBlock(512 + 1024, 512, self.up_sample_mode)
        self.up_conv3 = UpBlock(256 + 512, 256, self.up_sample_mode)
        self.up_conv2 = UpBlock(128 + 256, 128, self.up_sample_mode)
        self.up_conv1 = UpBlock(128 + 64, 64, self.up_sample_mode)
        # Final Convolution
        self.conv_last = nn.Conv2d(64, out_classes, kernel_size=1)

    def forward(self, x):
        x, skip1_out = self.down_conv1(x)
        x, skip2_out = self.down_conv2(x)
        x, skip3_out = self.down_conv3(x)
        x, skip4_out = self.down_conv4(x)
        x = self.double_conv(x)
        x = self.up_conv4(x, skip4_out)
        x = self.up_conv3(x, skip3_out)
        x = self.up_conv2(x, skip2_out)
        x = self.up_conv1(x, skip1_out)
        x = self.conv_last(x)
        return x

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

# def info(playback: PyK4APlayback):
#     print(f"Record length: {playback.length / 1000000: 0.2f} sec")
          
# Load an MKV file
# def create_frames(path: str):
#     # Load an MKV file
#     if os.path.exists(f"{path}"):
#         mkv_file = f"{path}"
#     else:
#         print("That path doesn't exist bruh lol")
#         exit()

#     playback = PyK4APlayback(mkv_file)
#     playback.open()
#     playback.seek(5000000) # in microseconds, aka 5 seconds
#     info(playback) # prints the recording length

#     # print(type(playback))
#     # Create a new HDF5 file
#     file = h5py.File('output_1.h5', 'w')

#     # Create datasets for RGB images and depth maps
#     rgb_images = file.create_dataset('images', (0, 640, 480, 3), maxshape=(None, 640,480, 3))
#     depth_images = file.create_dataset('depths', (0, 640, 480), maxshape=(None, 640, 480))

#     i = 0
#     while True:
#         try:
#             capture = playback.get_next_capture()

#             if capture.color is not None:
#                 img_color = cv2.resize(cv2.cvtColor(convert_to_bgra_if_required(0, capture.color), cv2.COLOR_BGR2RGB)[80:720, 446: 926, 0:3], (480, 640))

#                 # Append the RGB image to the dataset
#                 rgb_images.resize(i + 1, axis=0)
#                 rgb_images[i] = img_color

#             if capture.depth is not None:
#                 img_depth = cv2.resize(capture.transformed_depth[80:720, 446: 926], (480, 640))

#                 # Append the depth map to the dataset
#                 depth_images.resize(i + 1, axis=0)
#                 depth_images[i] = img_depth

#             i += 1

#             # key = cv2.waitKey(0)
#             # if key != -1:
#             #     break
#         except EOFError:
#             break

#     # Close the HDF5 file
#     file.close()

#     # Close the MKV file
#     playback.close()

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
    

def main(path: str):
    # create_frames(f"{path}")
    x, y = create_data_set("ecj1204.h5")
    loader = create_data_loader(x, y)
    print("done")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Now using device: {device}")
    model = UNet().to(torch.device(device))
    model.load_state_dict(torch.load('../../epoch_250.pt')['model_state_dict'])
    model.eval()
    print(f"Put model on device")
    with torch.no_grad():
        for batch_idx, (image, truth) in enumerate(loader):
            plt.figure()
            plt.subplot(1,3,1)
            color_img = image[0].transpose(0,1)
            color_img = color_img.transpose(1,2)
            color_img = torch.round(color_img).to(dtype=torch.int)
            plt.imshow(color_img)
            plt.title('Color image')
                        
            plt.subplot(1,3,2)
            plt.imshow(truth[0][0])
            plt.title('Ground Truth')
            
            start_time = time.perf_counter()
            image = image / 255.0
            image = image.to(torch.device(device))
            truth = truth.to(torch.device(device))
            cpu_time = time.perf_counter()
            
            outputs = model(image)
            print(f"Model ran for batch {batch_idx}")


            outputs = model(image)
            outputs = 1000.0 / outputs
            gpu_time = time.perf_counter()
            plt.subplot(1,3,3)
            plt.imshow(outputs[0][0].cpu())
            plt.title('Predicted')
            plt.show()

            total_time = gpu_time - start_time
            gpu_time = gpu_time - cpu_time
            cpu_time = cpu_time - start_time
            print(f"Model ran for batch {batch_idx}")
            print(f'CPU time: {cpu_time}, GPU time: {gpu_time}, FPS: {1/total_time}')
            break

def main2():
    # create_frames(f"{path}")
    #x, y = create_data_set("ecj1204.h5")
    #loader = create_data_loader(x, y)
    #print("done")
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Now using device: {device}")
    model = UNet().to(torch.device(device))
    model.load_state_dict(torch.load('../../epoch_250.pt')['model_state_dict'])
    model.eval()
    print(f"Model loaded!")
    # Create a PyK4A object
    print(f"Starting capture...")
    config = Config(
        color_resolution=ColorResolution.RES_720P,
        depth_mode=DepthMode.NFOV_UNBINNED,
        camera_fps=FPS.FPS_15
        #color_format=pyk4a.ImageFormat.COLOR_BGRA32,
    )

    k4a = PyK4A(config)

    # Open the device
    k4a.start()

    # Start the cameras using the default configuration
    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    while True:
        # Get a capture
        capture = k4a.get_capture()

        # If a capture is available
        if capture is not None:
            # Get the color image from the capture
            color_image = capture.color
            depth_image = capture.depth
            transformed_depth_image = capture.transformed_depth
            
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)[120:600, 320:960, 0:3]
            color_image_tensor = torch.from_numpy(color_image_rgb)
            color_image_tensor = color_image_tensor.transpose(0, 1).transpose(0, 2).contiguous()
            color_image_tensor = color_image_tensor.float().div(255)
            model_input = color_image_tensor.unsqueeze(0).to(device)
            #print(model_input)
            # model_input = convert_bgra_to_tensor(color_image)
            pred = model(model_input)
            pred = pred.detach().squeeze(0).squeeze(0).cpu()
            pred = 1000 / pred

            # If the 'q' key is pressed, break the loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
                # break
            axs[0].imshow(color_image_rgb)
            axs[0].set_title('Color Image')

            # Display the transformed depth image
            axs[1].imshow(transformed_depth_image)
            axs[1].set_title('Transformed Depth Image')

            # Display the predicted image
            axs[2].imshow(pred)
            axs[2].set_title('Predicted Image')



            plt.pause(0.001)  # Pause for a short period to allow the images to update

            # If the 'q' key is pressed, break the loop
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    # Stop the cameras and close the device
    k4a.device_stop_cameras()
    k4a.device_close()

    # Close the OpenCV window
    cv2.destroyAllWindows()

if __name__ == "__main__":
#    argsparser = argparse.ArgumentParser()
#    argsparser.add_argument("--path", help="path to mkv file")
#    args = argsparser.parse_args()
#    main(args.path)
     main2()
