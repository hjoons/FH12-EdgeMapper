'''from pyk4a import PyK4A, Config

# Create a PyK4A object
k4a = PyK4A(Config())

# Open the device
k4a.start()

# Start the cameras using the default configuration
capture = k4a.get_capture()

# Get a capture
# You can access the color, depth, and IR images like this:
color_image = capture.color
depth_image = capture.depth
ir_image = capture.ir

# Don't forget to stop the cameras and close the device
k4a.stop()'''
# import cv as mkv_reader
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
    

def convert_rgb_to_tensor(image):
    resized_image = cv2.resize(image, (480, 640))

    bgr_image = resized_image[:, :, :3]
    tensor_image = torch.from_numpy(bgr_image).permute(2, 0, 1).float()
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def convert_bgra_to_bgr(image):
    resized_image = cv2.resize(image, (480, 640))

    bgr_image = resized_image[:, :, :3]

    return bgr_image

def convert_depth_to_tensor(image):
    resized_image = cv2.resize(image, (480, 640))

    depth_image = resized_image[:, :, :1]
    tensor_image = torch.from_numpy(depth_image).permute(2, 0, 1).float()
    tensor_image = tensor_image.unsqueeze(0)

    return tensor_image

def main():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")
    print(f"Loading model...")
    model = UNet().to(torch.device(device))
    model = model.load_state_dict(torch.load('../../epoch_250.pt')['model_state_dict'])
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
    fig, axs = plt.subplots(1, 2, figsize=(20, 7))

    while True:
        # Get a capture
        capture = k4a.get_capture()

        # If a capture is available
        if capture is not None:
            # Get the color image from the capture
            color_image = capture.color
            depth_image = capture.depth
            transformed_depth_image = capture.transformed_depth
            
            color_image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGRA2RGB)
            model_input = color_image_rgb.unsqueeze(0)
            # model_input = convert_bgra_to_tensor(color_image)
            pred = model(model_input)
            pred = pred.squeeze(0).squeeze(0).cpu()
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

main()
