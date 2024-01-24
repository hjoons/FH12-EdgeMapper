import torch
import numpy as np
from PIL import Image
import torchsummary

def threshold_percentage(output, target, threshold_val):
    """
    Function that calculates the percentage of pixels that have a ratio between the output and target
    that is less than a threshold value

    Args:
        output (torch.Tensor): Output tensor
        target (torch.Tensor): Target tensor
        threshold_val (float): Threshold value

    Returns:
        float: Percentage of pixels that have a ratio between the output and target that is less than a threshold value
    """
    # Scale invariant
    
    d1 = output / target
    d2 = target / output
    
    max_d1_d2 = torch.max(d1, d2)
    
    zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    bit_mat = torch.where(max_d1_d2.cpu() < threshold_val, one, zero)
    count_mat = torch.sum(bit_mat, (1, 2, 3))
    threshold_mat = count_mat / (output.shape[2] * output.shape[3])
    return threshold_mat.mean()

def REL(output, target):
    """
    Function that calculates the relative error between the output and target

    Args:
        output (torch.Tensor): Output tensor
        target (torch.Tensor): Target tensor

    Returns:
        float: Relative error between the output and target
    """
    diff = torch.abs(target - output) / target
    return torch.sum(diff) / (output.shape[2] * output.shape[3])

def RMS(output, target):
    """
    Function that calculates the root mean squared error between the output and target

    Args:
        output (torch.Tensor): Output tensor
        target (torch.Tensor): Target tensor

    Returns:
        float: Root mean squared error between the output and target
    """
    diff = target - output
    squared = torch.square(diff)
    summed = torch.sum(squared) / (output.shape[2] * output.shape[3])
    return torch.sqrt(summed)

def evaluate(model, test_loader):
    """
    Function that evaluates the model on the test set. Calculates the deltas, REL, and RMS

    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): Test loader

    Returns:
        None
    """
    model.eval()
    d1 = 0
    d2 = 0
    d3 = 0
    rel = 0
    rms = 0
    for batchidx, batch in enumerate(test_loader):
        print(f"{batchidx + 1} / {len(test_loader)}")
        inputs, targets = batch
        with torch.no_grad():
            outputs = model(inputs)

            d1 += threshold_percentage(outputs, targets, 1.25)
            d2 += threshold_percentage(outputs, targets, 1.5625)
            d3 += threshold_percentage(outputs, targets, 1.953125)

            rel += REL(outputs, targets)
            
            rms += RMS(outputs, targets)

    d1 = d1 / len(test_loader)
    d2 = d2 / len(test_loader)
    d3 = d3 / len(test_loader)
    rel = (rel / len(test_loader)).item()
    rms = (rms / len(test_loader)).item()
    deltas = (d1.item(), d2.item(), d3.item())
    print(f"deltas: {deltas}")
    print(f"REL: {rel}") # Not working correctly
    print(f"RMS: {rms}")


def DepthNorm(depth, max_depth=1000.0):
    return max_depth / depth

def load_images(image_files):
    """
    Function that loads images from a list of image files

    Args:
        image_files (list): List of image files

    Returns:
        np.ndarray: Array of images
    """
    loaded_images = []
    for file in image_files:
        x = np.clip(
            np.asarray(Image.open(file).resize((640, 480)), dtype=float) / 255, 0, 1
        ).transpose(2, 0, 1)

        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)

def model_summary(model):
    """
    Function that prints the model summary

    Args:
        model (torch.nn.Module): Model to print summary of

    Returns:
        None
    """
    model = model.to(torch.device("cuda"))
    torchsummary.summary(model, input_size=(3,640,480), batch_size=1)