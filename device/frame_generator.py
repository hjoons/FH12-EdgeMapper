import matplotlib.pyplot as plt
import cv2
from pyk4a import PyK4A, Config, FPS, DepthMode, ColorResolution
import os
import h5py
from torchvision import transforms
from torch.utils.data import DataLoader
import time
import torch
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.unet import UNet
from models.mobilenetv3 import MobileNetSkipConcat
from dataset.dataset import CustomDataset




def create_data_set(file_name):
    """
    Creates a list of input training tensors and output training tensors from a .h5 file.
    Normalizes the output tensors by dividing by the max value of the output tensors.

    Args:
        file_name (str): name of the .h5 file

    Returns:
        X_train_tensors (list): list of input training tensors
        y_train_tensors (list): list of output training tensors
    """
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
    """
    Creates a data loader from a list of input tensors and output tensors

    Args:
        x (list): list of input tensors
        y (list): list of output tensors
    
    Returns:
        loader (torch.utils.data.DataLoader): data loader
    """
    dataset = CustomDataset(x, y)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)
    return loader
    

def main():
    # create_frames(f"{path}")
    # x, y = create_data_set("ecj1204.h5")

    # Create a data loader from an h5 file
    print("Creating data loader...")
    x, y = create_data_set("C:/Users/vliew/Documents/UTAustin/Fall2023/SeniorDesign/FH12-EdgeMapper/Device1/orgspace.h5")
    loader = create_data_loader(x, y)
    print("done")
    print()

    # Choose the device to run the model on, and load the model into memory
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Now using device: {device}")
    model = MobileNetSkipConcat().to(torch.device(device))
    model.load_state_dict(torch.load('../../mbnv3_epoch_100.pt', map_location=torch.device(device))['model_state_dict'])
    model.eval()
    print(f"Put model on device")

    fig, axs = plt.subplots(1, 3, figsize=(20, 7))

    with torch.no_grad():
        for batch_idx, (image, truth) in enumerate(loader):
            # Convert input from (batch_size, 3, 480, 640) to (batch_size, 480, 640, 3)
            color_img = image[0].transpose(0,1)
            color_img = color_img.transpose(1,2)
            color_img = torch.round(color_img).to(dtype=torch.int)
            
            start_time = time.perf_counter()
            # Convert to tensor in range [0, 1]
            image = image / 255.0
            image = image.to(torch.device(device))
            # truth = truth.to(torch.device(device))
            cpu_time = time.perf_counter()

            outputs = model(image)
            outputs = 1000.0 / outputs
            gpu_time = time.perf_counter()
            print(f"Model ran for batch {batch_idx}")

            axs[0].imshow(color_img)
            axs[0].set_title('Color Image')

            # Display the transformed depth image
            axs[1].imshow(truth[0][0])
            axs[1].set_title('Transformed Depth Image')

            # Display the predicted image
            axs[2].imshow(outputs[0][0])
            axs[2].set_title('Predicted Image')

            plt.pause(0.001)  # Pause for a short period to allow the images to update

            total_time = gpu_time - start_time
            gpu_time = gpu_time - cpu_time
            cpu_time = cpu_time - start_time
            # print(f"Model ran for batch {batch_idx}")
            print(f'CPU time: {cpu_time}, GPU time: {gpu_time}, FPS: {1/total_time}')
            #break

def main2():
    # Choose the device to run the model on, and load the model into memory
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
        color_resolution=ColorResolution.RES_720P, # 720P to balance quality and speed
        depth_mode=DepthMode.NFOV_UNBINNED, # Narrow Field of View because we crop, unbinned to retain resolution
        camera_fps=FPS.FPS_15 # 15 Frames per Second
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
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

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
     main()
