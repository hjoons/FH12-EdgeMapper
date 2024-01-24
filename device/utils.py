import paramiko
import socket
from scp import SCPClient
import zipfile
import os
import torch
from typing import Optional, Tuple
from pyk4a import ImageFormat
import cv2
import numpy as np
import sys

# Add parent directory to path to be able to import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import dataset.h5dataset as h5dataset
from models.unet import UNet
from models.mobilenetv3 import MobileNetSkipConcat
from models.losses import ssim, depth_loss

def convert_to_bgra_if_required(color_format: ImageFormat, color_image):
    """
    Converts the color image to BGRA if required.

    Args:
        color_format (ImageFormat): Color format of the image
        color_image (np.ndarray): Color image

    Returns:
        np.ndarray: Color image in BGRA format
    """
    # examples for all possible pyk4a.ColorFormats
    if color_format == ImageFormat.COLOR_MJPG:
        color_image = cv2.imdecode(color_image, cv2.IMREAD_COLOR)
    elif color_format == ImageFormat.COLOR_NV12:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_NV12)
        # this also works and it explains how the COLOR_NV12 color color_format is stored in memory
        # h, w = color_image.shape[0:2]
        # h = h // 3 * 2
        # luminance = color_image[:h]
        # chroma = color_image[h:, :w//2]
        # color_image = cv2.cvtColorTwoPlane(luminance, chroma, cv2.COLOR_YUV2BGRA_NV12)
    elif color_format == ImageFormat.COLOR_YUY2:
        color_image = cv2.cvtColor(color_image, cv2.COLOR_YUV2BGRA_YUY2)
    return color_image


def colorize(
    image: np.ndarray,
    clipping_range: Tuple[Optional[int], Optional[int]] = (None, None),
    colormap: int = cv2.COLORMAP_HSV,
) -> np.ndarray:
    if clipping_range[0] or clipping_range[1]:
        img = image.clip(clipping_range[0], clipping_range[1])  # type: ignore
    else:
        img = image.copy()
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    img = cv2.applyColorMap(img, colormap)
    return img

def DepthNorm(depth, max_depth=1000.0):
    """
    Normalize depth values. The depth values are transformed by dividing the maximum depth value by the depth value so that depth values 
    that there is more granularity in the smaller depth values.

    Args:
        depth (float): Depth value to be normalized.
        max_depth (float): Maximum depth value for normalization.

    Returns:
        float: Normalized depth value.
    """
    return max_depth / depth

def zip_file(file, zip_file_name):
    """
    Zip a file

    Args:
        file (str): path of the file to be zipped - *MUST be relative path
        zip_file_name (str): path of the zipped file

    Returns:
        None
    """
    try:
        with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file, arcname=file)
        # print(f'{file} has been zipped to {zip_file_name}')
    except Exception as e:
        print(f'Error zipping file: {e}')

def unzip_file(zip_file, destination_folder):
    """
    Unzip a file

    Args:
        zip_file (str): path of the zip file to unzip
        destination_folder (str): path to place the unzipped file

    Returns:
        None
    """
    try:
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall(destination_folder)
        # print(f'{zip_file} has been successfully unzipped to {destination_folder}')
    except Exception as e:
        print(f'Error unzipping file: {e}')
      

def train(learning_rate, num_epochs, file_path, federated_path):
    """
    Train the model using federated learning.

    Args:
        learning_rate (float): Learning rate for the optimizer.
        num_epochs (int): Number of training epochs.
        file_path (str): Path to the pretrained model file.
        federated_path (str): Path to save the federated model.

    Returns:
        str: Path to the saved federated model.
    """
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Now using device: {device}")

    # model initialization and loading weights
    model = MobileNetSkipConcat().to(torch.device(device))
    model.load_state_dict(torch.load(f'{file_path}', map_location=torch.device(device))['model_state_dict'])

    # data loading
    train_loader, val_loader = h5dataset.createH5TrainLoader(path='C:/Users/vliew/Documents/UTAustin/Fall2023/SeniorDesign/FH12-EdgeMapper/Device2/eer-ecj.h5', batch_size=1)

    # custom_loader = dataset.createCustomDataLoader(f'{file_path}')
    print("Custom loader len: ", len(train_loader))
    print("DataLoaders now ready ...")

    num_trainloader = len(train_loader)

    # optimizer and loss criterion optimization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    l1_criterion = torch.nn.L1Loss()
    train_loss = []
    test_loss = []
    print(f"About to train")

    # training loop
    for epoch in range(num_epochs):
        # time_start = time.perf_counter()
        model = model.train()
        running_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            image = torch.Tensor(batch['image']).to(device)
            depth = torch.Tensor(batch['depth']).to(device)

            # print(f"putting images on device")
            #image = image.to(device)
            #depth = truth.to(device)

            normalized_depth = DepthNorm(depth)
            
            pred = model(image)

            l1_loss = l1_criterion(pred, normalized_depth)
            
            ssim_loss = torch.clamp(
                (1 - ssim(pred, normalized_depth, 1000.0 / 10.0)) * 0.5,
                min=0,
                max=1,
            )
            
            gradient_loss = depth_loss(normalized_depth, pred, device=device)
            
            net_loss = (
                (1.0 * ssim_loss)
                + (1.0 * torch.mean(gradient_loss))
                + (0.1 * torch.mean(l1_loss))
            )
            
            cpu_loss = net_loss.cpu().detach().numpy()
            # writer.add_scalar("Loss/batch_train",cpu_loss,batch_iter)
            running_loss += cpu_loss

            net_loss.backward()

            optimizer.step()
            break # Only for demo purposes

        train_loss.append(running_loss / num_trainloader)
        print(f'epoch: {epoch + 1} train loss: {running_loss / num_trainloader}')

    # save trained model
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    model_path = os.path.join(os.getcwd(), federated_path)
    torch.save(checkpoint, model_path)
    return federated_path

# Function to send a message over a socket
def send_message(sock: socket.socket, msg: str):
    """
    Function to send a message over a socket

    Args:
        sock (socket): Socket over which messages are sent and received.
        msg (str): Message to be sent.

    Returns:
        None
    """
    sock.sendall(msg.encode())

# Function to receive a message over a socket
def receive_message(sock: socket.socket) -> str:
    """
    Function to receive a message over a socket.

    Args:
        sock (socket): Socket over which messages are sent and received.

    Returns:
        str: Received message.
    """
    msg = sock.recv(1024)
    return msg.decode()

def send_scp_file(local_path: str, remote_path: str, ip: str, user: str, pwd: str, file_name: str, sock: socket.socket, zip_name: str):
    """
    Send a file using SCP.

    Args:
        local_path (str): Local path of the file.
        remote_path (str): Remote path to save the file.
        ip (str): IP address of the remote server.
        user (str): Username for authentication.
        pwd (str): Password for authentication.
        file_name (str): Name of the file to be sent.
        sock (socket): Socket over which messages are sent and received.
        zip_name (str): Name of the zipped file.
    Returns:
        None
    """
    # zip_file(local_path + file_name, local_path + zip_name)
    file_size = os.path.getsize(local_path + file_name)
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=ip, username=user, password=pwd, banner_timeout=200)

    received = False
    while not received:
        try:
            with SCPClient(ssh.get_transport()) as scp:
                scp.put(local_path + zip_name, remote_path + zip_name)
                # print(f"Sent file: {local_path + zip_name}")
        except Exception as e:
            print(f"Error sending file: {e}")
            continue
        
        msg = ";".join([remote_path + zip_name, local_path + file_name, str(file_size)])
        send_message(sock, msg)

        msg = receive_message(sock)
        # print(msg)
        if msg == 'Received':
            received = True

    
    ssh.close()

def receive_scp_file(destination_path: str, sock: socket.socket):
    """
    Receive file through SCP

    Waits for incoming message with the name of the zip file, name of the unzipped file,
    and the size of the file. Checks file size until correct file size received.

    Args:
        destination_path (str): path to place the unzipped file
        sock (socket): socket over which messages are sent and received

    Returns:
        str: path of the received file
    """
    received = False
    file_name = ""

    while not received:
        msg = receive_message(sock)
        # print(msg)
        split_msg = msg.split(';')
        zip_locale = os.path.expanduser(split_msg[0])
        file_name = split_msg[1]
        sent_file_size = int(split_msg[2])

        if os.path.exists(zip_locale):
            unzip_file(zip_locale, destination_path)

            local_file_size = os.path.getsize(destination_path + file_name)
            if local_file_size == sent_file_size:
                received = True
                send_message(sock, "Received")
            else:
                send_message(sock, "Resend")

            # delete zip file
            os.remove(zip_locale)
        else:
            send_message(sock, "Resend")

    return destination_path + file_name
