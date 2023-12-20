import paramiko
import socket
import h5py
from scp import SCPClient
import zipfile
import os
import torch
import random
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from models.unet import UNet
from models.mobilenetv3 import MobileNetSkipConcat
import dataset.h5dataset as h5dataset
from models.losses import ssim, depth_loss
import copy

def DepthNorm(depth, max_depth=1000.0):
    """
    Normalize depth values.

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

# Function to send a message over a socket
def send_message(sock: socket.socket, msg: str):
    sock.sendall(msg.encode())

# Function to receive a message over a socket
def receive_message(sock: socket.socket) -> str:
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

def fed_avg(models_dir:str):
    """
    Perform federated averaging on the models in the specified directory and save the federated model.

    Args:
        models_dir (str): Path to the directory containing the models to be averaged.

    Returns:
        None
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # get all the models in the models_dir directory
    models = os.listdir(models_dir)
    num_models = len(models)
    print(f"Number of models: {num_models}")

    # load all the weights
    loaded_weights = []
    for model in models:
        local_weight = torch.load(models_dir + model, map_location=torch.device(device))['model_state_dict']
        loaded_weights.append(local_weight)

    # calculate the federated average of the weights
    checkpoint = {
        'model_state_dict': aggregate_avg(loaded_weights)
    }

    # save global model
    if os.path.exists('global_model.pt'):
        os.remove('global_model.pt')
    torch.save(checkpoint, 'global_model.pt')

def aggregate_avg(local_weights):
    """
    Aggregate and calculate the average of model weights.

    Args:
        local_weights: list of local model weights

    Returns:
        Dict[str, torch.Tensor]: Average of the model weights.
    """
    w_glob = None
    for idx, w_local in enumerate(local_weights):
        if idx == 0:
            w_glob = copy.deepcopy(w_local)
        else:
            for k in w_glob.keys():
                w_glob[k] = torch.add(w_glob[k], w_local[k])
    for k in w_glob.keys():
        w_glob[k] = torch.div(w_glob[k], len(local_weights))
    return w_glob

def federated_averaging(models_dir: str):
    """
    Perform federated averaging on the models in the models_dir directory

    Args:
        models_dir (str): path to the directory containing the models to be averaged

    Returns:
        None
    """

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    # get all the models in the models_dir directory
    models = os.listdir(models_dir)
    num_models = len(models)
    print(f"Number of models: {num_models}")

    # load all the models
    loaded_models = []
    for model in models:
        unet = MobileNetSkipConcat()
        unet.load_state_dict(torch.load(models_dir + model, map_location=torch.device(device))['model_state_dict'])
        unet.eval()
        loaded_models.append(unet)
    
    federated_model = MobileNetSkipConcat()
    federated_model.eval()

    # perform federated averaging
    for param_fm, *params_m in zip(federated_model.parameters(), *[model.parameters() for model in loaded_models]):
        param_fm.data.copy_(torch.stack([param_m.data for param_m in params_m]).mean(0))

    checkpoint = {
        'model_state_dict': federated_model.state_dict(),
    }
    if os.path.exists('global_model.pt'):
        os.remove('global_model.pt')
    torch.save(checkpoint, 'global_model.pt')

def compute_errors(gt, pred, epsilon=1e-6):
    """
    Compute error metrics between ground truth and predicted depth maps.

    Args:
        gt (torch.Tensor): Ground truth depth map.
        pred (torch.Tensor): Predicted depth map.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        Tuple: Tuple containing various error metrics.
    """
    # Ensure non-zero and non-negative ground truth values
    gt = gt.float().to('cpu')
    pred = pred.float().to('cpu')

    gt = torch.clamp(gt, min=epsilon)
    pred = torch.clamp(pred, min=epsilon)  # Also ensure predictions are positive

    thresh = torch.max((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < (1.25 ** 2)).float().mean()
    a3 = (thresh < (1.25 ** 3)).float().mean()

    rmse = torch.sqrt(((gt - pred) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log(gt) - torch.log(pred)) ** 2).mean())

    abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    sq_rel = torch.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def federated_eval(eval_dir: str):
    """
    Evaluate a federated model on a dataset.

    Args:
        eval_dir (str): Path to the dataset for evaluation.

    Returns:
        None
    """
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
    )
    model = MobileNetSkipConcat().to(device)
    model.load_state_dict(torch.load('global_model.pt', map_location=torch.device(device))['model_state_dict'])
    model.eval()
    
    f = h5py.File(eval_dir)
    errors = []

    random_indices = set()
    while len(random_indices) < 50:
        random_indices.add(random.randint(0, len(f['images']) - 1))

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(list(random_indices)):
            # print(f'{i + 1} / 100')
            img = f['images'][idx].astype(float)
            img = torch.Tensor(img) / 255.0
            img = img.permute(2, 1, 0)

            gt = f['depths'][idx]
            gt = torch.Tensor(gt)
            gt = gt.unsqueeze(0).permute(0, 2, 1)
            gt = gt / 1000.0
            gt = torch.clamp(gt, 10, 1000)

            input_tensor = img.unsqueeze(0)

            pred = model(input_tensor)
            pred = pred.squeeze(0)

            errors.append(compute_errors(gt, pred))

        error_tensors = [torch.tensor(e).to(device) for e in errors]

        error_stack = torch.stack(error_tensors, dim=0)

        mean_errors = error_stack.mean(0).cpu().numpy()

        abs_rel = mean_errors[0]
        sq_rel = mean_errors[1]
        rmse = mean_errors[2]
        rmse_log = mean_errors[3]
        a1 = mean_errors[4]
        a2 = mean_errors[5]
        a3 = mean_errors[6]
        print(f'abs_rel: {abs_rel}\nsq_rel: {sq_rel}\nrmse: {rmse}\nrmse_log: {rmse_log}\na1: {a1}\na2: {a2}\na3: {a3}\n')

if __name__ == '__main__':
    print("utils.py")