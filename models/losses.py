import torch
import torch.nn.functional as F
import math
import numpy as np

""" Loss file implementation referred from 
https://github.com/ialhashim/DenseDepth/blob/master/PyTorch/loss.py
"""

def gaussian(window_size, sigma):
    """
    Function that returns a 1-dimensional Gaussian window of a given size and sigma (standard deviation)

    Args:
        window_size (int): Size of the window
        sigma (float): Standard deviation of the Gaussian

    Returns:
        torch.Tensor: 1-dimensional Gaussian window
    """
    gauss = torch.Tensor(
        [
            math.exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2))
            for x in range(window_size)
        ]
    )
    return gauss / gauss.sum()


def create_window(window_size, channel=1):
    """
    Function that creates a 2-dimensional Gaussian window of a given size and sigma (standard deviation)

    Args:
        window_size (int): Size of the window
        channel (int, optional): Number of channels. Defaults to 1.

    Returns:
        torch.Tensor: 2-dimensional Gaussian window expanded to the number of channels
    """

    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)

    window = torch.Tensor(
        _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    )

    return window


def ssim(
    img1, img2, val_range, window_size=11, window=None, size_average=True, full=False
):
    """
    Function that calculates the Structural Similarity Index Measure (SSIM) loss between two images

    Args:
        img1 (torch.Tensor): First image
        img2 (torch.Tensor): Second image
        val_range (int): Range of pixel values
        window_size (int, optional): Size of the window. Defaults to 11.
        window (torch.Tensor, optional): Window tensor. Defaults to None.
        size_average (bool, optional): If True, the loss is averaged over all images in the batch. Defaults to True.
        full (bool, optional): If True, returns the full SSIM instead of the mean SSIM. Defaults to False.

    Returns:
        torch.Tensor: SSIM loss
    """

    L = val_range  # L is the dynamic range of the pixel values (255 for 8-bit grayscale images),

    pad = window_size // 2

    try:
        _, channels, height, width = img1.size()
    except:
        channels, height, width = img1.size()

    # if window is not provided, init one
    if window is None:
        real_size = min(window_size, height, width)  # window should be atleast 11x11
        window = create_window(real_size, channel=channels).to(img1.device)

    # calculating the mu parameter (locally) for both images using a gaussian filter
    # calculates the luminosity params
    mu1 = F.conv2d(img1, window, padding=pad, groups=channels)
    mu2 = F.conv2d(img2, window, padding=pad, groups=channels)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu12 = mu1 * mu2

    # now we calculate the sigma square parameter
    # Sigma deals with the contrast component
    sigma1_sq = F.conv2d(img1 * img1, window, padding=pad, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=pad, groups=channels) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=pad, groups=channels) - mu12

    # Some constants for stability
    C1 = (0.01) ** 2  # NOTE: Removed L from here (ref PT implementation)
    C2 = (0.03) ** 2

    contrast_metric = (2.0 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2)
    contrast_metric = torch.mean(contrast_metric)

    numerator1 = 2 * mu12 + C1
    numerator2 = 2 * sigma12 + C2
    denominator1 = mu1_sq + mu2_sq + C1
    denominator2 = sigma1_sq + sigma2_sq + C2

    ssim_score = (numerator1 * numerator2) / (denominator1 * denominator2)

    if size_average:
        ret = ssim_score.mean()
    else:
        ret = ssim_score.mean(1).mean(1).mean(1)

    if full:
        return ret, contrast_metric

    return ret


def image_gradients(img, device):
    """
    Function that calculates the image gradients in both directions of a given image

    Args:
        img (torch.Tensor): Image
        device (str): Device to use

    Returns:
        torch.Tensor: Image gradients in the y direction
        torch.Tensor: Image gradients in the x direction
    """

    """works like tf one"""
    if len(img.shape) != 4:
        raise ValueError("Shape mismatch. Needs to be 4 dim tensor")

    img_shape = img.shape
    batch_size, channels, height, width = img.shape
        
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]

    shape = np.stack([batch_size, channels, 1, width])
    dy = torch.cat(
        [
            dy,
            torch.zeros(
                [batch_size, channels, 1, width], device=device, dtype=img.dtype
            ),
        ],
        dim=2,
    )
    dy = dy.view(img_shape)

    shape = np.stack([batch_size, channels, height, 1])
    dx = torch.cat(
        [
            dx,
            torch.zeros(
                [batch_size, channels, height, 1], device=device, dtype=img.dtype
            ),
        ],
        dim=3,
    )
    dx = dx.view(img_shape)

    return dy, dx


# Now we define the actual depth loss function
def depth_loss(y_true, y_pred, theta=0.1, device="cuda", maxDepth=1000.0 / 10.0):
    """
    Depth loss function:
        - Takes image gradients and computes the depth loss using L1 norm

    Args:
        y_true (torch.Tensor): Ground truth depth map
        y_pred (torch.Tensor): Predicted depth map
        theta (float, optional): Constant to scale the depth loss. Defaults to 0.1.
        device (str, optional): Device to use. Defaults to "cuda".
        maxDepth (float, optional): Maximum depth. Defaults to 100.0.

    Returns:
        torch.Tensor: Depth loss
    """

    # Edges
    dy_true, dx_true = image_gradients(y_true, device)
    dy_pred, dx_pred = image_gradients(y_pred, device)
    l_edges = torch.mean(
        torch.abs(dy_pred - dy_true) + torch.abs(dx_pred - dx_true), dim=1
    )

    return l_edges

def custom_loss(pred, target):
    di = target - pred
    n = 640*480
    di2 = torch.pow(di, 2)
    first_term = torch.sum(di2, (1, 2, 3)) / n
    second_term = .5 * torch.pow(torch.sum(di, (1, 2, 3)), 2) / (n**2)
    loss = first_term - second_term
    return loss.mean()

def silog(pred, target, delta=.5):
    """
    Scale Invariant Log Loss
    https://papers.nips.cc/paper/2014/file/7bccfde7714a1ebadf06c5f4cea752c1-Paper.pdf

    Args:
        pred (torch.Tensor): Predictions.
        target (torch.Tensor): Ground truth.
        delta (float, optional): Delta that goes along with literature. Defaults to .5.

    Returns:
        float: SILog Loss. 
    """
    mask = (pred > 0) & (target > 0)
    n = len(torch.nonzero(mask))
    
    d = torch.log(pred[mask]) - torch.log(target[mask])
    loss = (1 / n) * (torch.sum(d ** 2)) - (delta / (n ** 2)) * (torch.sum(d) ** 2)
    return loss