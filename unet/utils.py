import torch
import numpy as np
from PIL import Image

def threshold_percentage(output, target, threshold_val):
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
    diff = torch.abs(target - output) / target
    return torch.sum(diff) / (output.shape[2] * output.shape[3])

def RMS(output, target):
    diff = target - output
    squared = torch.square(diff)
    summed = torch.sum(squared) / (output.shape[2] * output.shape[3])
    return torch.sqrt(summed)

def evaluate(model, test_loader):
    model.eval()
    d1 = 0
    d2 = 0
    d3 = 0
    for batchidx, batch in enumerate(test_loader):
        print(f"{batchidx + 1} / {len(test_loader)}")
        inputs, targets = batch
        with torch.no_grad():
            outputs = model(inputs)

            d1 += threshold_percentage(outputs, targets, 1.25)
            d2 += threshold_percentage(outputs, targets, 1.5625)
            d3 += threshold_percentage(outputs, targets, 1.953125)
    
    d1 = d1 / len(test_loader)
    d2 = d2 / len(test_loader)
    d3 = d3 / len(test_loader)
    deltas = (d1, d2, d3)
    print(deltas)

def DepthNorm(depth, max_depth=1000.0):
    return max_depth / depth

def load_images(image_files):
    loaded_images = []
    for file in image_files:
        x = np.clip(
            np.asarray(Image.open(file).resize((640, 480)), dtype=float) / 255, 0, 1
        ).transpose(2, 0, 1)

        loaded_images.append(x)
    return np.stack(loaded_images, axis=0)