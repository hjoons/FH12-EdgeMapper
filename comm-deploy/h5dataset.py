import h5py

from PIL import Image
from dataset import getDefaultTrainTransform, getNoTransform
from torch.utils.data import DataLoader, Dataset


class H5DepthDataset(Dataset):
    def __init__(self, h5_path, transform=None):
        h5_file = h5py.File(h5_path, 'r')
        self.h5_file = h5_file
        self.transform = transform
        
    def __len__(self):
        return len(self.h5_file['images'])
    
    def __getitem__(self, index):
        img = self.h5_file['images'][index]
        gt = self.h5_file['depths'][index]

        pil_img = Image.fromarray(img, 'RGB')
        pil_gt = Image.fromarray(gt, 'L')

        sample = {"image": pil_img, "depth": pil_gt}
    
        if self.transform:
            sample = self.transform(sample)
            
        return sample
    
def createH5TrainLoader(path, batch_size=1):
    transformed_training = H5DepthDataset(path, transform=getDefaultTrainTransform())
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    
    return DataLoader(transformed_training, batch_size=batch_size, shuffle=True), DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)

def createH5TestLoader(path, batch_size=1):
    transformed_testing = H5DepthDataset(path, transform=getNoTransform())
    return DataLoader(transformed_testing, batch_size=batch_size, shuffle=False)
