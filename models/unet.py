import torch
from torch import nn

class DoubleConv(nn.Module):
    """
    Double Convolution Layer - (Conv2d + BatchNorm2d + ReLU) x 2
    """
    def __init__(self, in_channels, out_channels):
        """
        Constructor for DoubleConv class

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels

        Returns:
            None
        """
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
        """
        Perform forward pass of DoubleConv

        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor
        """
        return self.double_conv(x)


class DownBlock(nn.Module):
    """
    Down Block - DoubleConv + MaxPool2d performs Doubnle Convolution then downsamples by MaxPool2d
    """
    def __init__(self, in_channels, out_channels):
        """
        Constructor for DownBlock class

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels

        Returns:
            None
        """
        super(DownBlock, self).__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.down_sample = nn.MaxPool2d(2)

    def forward(self, x):
        """
        Perform forward pass of DownBlock

        Args:
            x (torch.Tensor): input tensor

        Returns:
            skip_out (torch.Tensor): output tensor from DoubleConv (skip connection used in UpBlock)
            down_out (torch.Tensor): output tensor from MaxPool2d
        """
        skip_out = self.double_conv(x)
        down_out = self.down_sample(skip_out)
        return (down_out, skip_out)


class UpBlock(nn.Module):
    """
    Up Block - UpSample + DoubleConv performs UpSample then Double Convolution
    """
    def __init__(self, in_channels, out_channels, up_sample_mode):
        """
        Constructor for UpBlock class

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            up_sample_mode (str): up sampling mode (can take one of `conv_transpose` or `bilinear`)

        Returns:
            None
        """
        super(UpBlock, self).__init__()
        if up_sample_mode == 'conv_transpose':
            self.up_sample = nn.ConvTranspose2d(in_channels-out_channels, in_channels-out_channels, kernel_size=2, stride=2)
        elif up_sample_mode == 'bilinear':
            self.up_sample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            raise ValueError("Unsupported `up_sample_mode` (can take one of `conv_transpose` or `bilinear`)")
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, down_input, skip_input):
        """
        Perform forward pass of UpBlock

        Args:
            down_input (torch.Tensor): input tensor from previous layer
            skip_input (torch.Tensor): input tensor from skip connection from the down sampling path

        Returns:
            torch.Tensor: output tensor. Concatenation of up sampled tensor and skip connection tensor that is passed through DoubleConv
        """
        x = self.up_sample(down_input)
        x = torch.cat([x, skip_input], dim=1)
        return self.double_conv(x)


class UNet(nn.Module):
    """
    UNet: Convolutional Neural Network for semantic segmentation
        - Downsampling Path: 
            - DownBlock x 4; channels: 3 -> 64 -> 128 -> 256 -> 512
        - Bottleneck:
            - DoubleConv; channels: 512 -> 1024
        - Upsampling Path:
            - UpBlock x 4 (with skip connections from Downsampling Path); channels: 512+1024 -> 512 -> 256 -> 128 -> 64
        - Final Convolution:
            - Conv2d; channels: 64 -> 1
    """
    def __init__(self, out_classes=1, up_sample_mode='conv_transpose'):
        """
        Constructor for UNet class

        Args:
            out_classes (int): number of output classes
            up_sample_mode (str): up sampling mode (can take one of `conv_transpose` or `bilinear`)
        
        Returns:
            None
        """
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
        """
        Perform forward pass of UNet

        Args:
            x (torch.Tensor): input tensor
        
        Returns:
            torch.Tensor: output tensor
        """
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