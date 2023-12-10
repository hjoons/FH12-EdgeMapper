import paramiko
import socket
import h5py
from scp import SCPClient
import zipfile
import os
import time
import h5dataset
import torch
import random
from frame_generator import UNet
from mobilenetv3 import MobileNetSkipConcat
from losses import ssim, depth_loss

def DepthNorm(depth, max_depth=1000.0):
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
    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
        )
    print(f"Now using device: {device}")
    model = MobileNetSkipConcat().to(torch.device(device))
    model.load_state_dict(torch.load(f'{file_path}', map_location=torch.device(device))['model_state_dict'])
    # model.eval()
    # not using any model information for this step
        # print(f"\tEpoch{i}")
        # # adding 1 second time delay
        # time.sleep(1)
    train_loader, val_loader = h5dataset.createH5TrainLoader(path='C:/Users/vliew/Documents/UTAustin/Fall2023/SeniorDesign/FH12-EdgeMapper/Device2/eer-ecj.h5', batch_size=1)

    # custom_loader = dataset.createCustomDataLoader(f'{file_path}')
    print("Custom loader len: ", len(train_loader))
    
    print("DataLoaders now ready ...")
    num_trainloader = len(train_loader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    l1_criterion = torch.nn.L1Loss()
    train_loss = []
    test_loss = []
    print(f"About to train")
    for epoch in range(num_epochs):
        time_start = time.perf_counter()
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
            # batch_iter += 1
            break # Only for demo purposes

        train_loss.append(running_loss / num_trainloader)
        print(f'epoch: {epoch + 1} train loss: {running_loss / num_trainloader}')
    checkpoint = {
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    model_path = os.path.join(os.getcwd(), federated_path)
    torch.save(checkpoint, model_path)
    return federated_path

def send_message(sock: socket.socket, msg: str):
    sock.sendall(msg.encode())

def receive_message(sock: socket.socket) -> str:
    msg = sock.recv(1024)
    return msg.decode()

def send_scp_file(local_path: str, remote_path: str, ip: str, user: str, pwd: str, file_name: str, sock: socket.socket, zip_name: str):
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

def federated_averaging(models_dir: str):
    """
    Perform federated averaging on the models in the models_dir directory

    Args:
        models_dir (str): path to the directory containing the models to be averaged

    Returns:
        None
    """
    # get all the models in the models_dir directory
    models = os.listdir(models_dir)
    num_models = len(models)
    print(f"Number of models: {num_models}")

    # load all the models
    loaded_models = []
    for model in models:
        unet = MobileNetSkipConcat()
        unet.load_state_dict(torch.load(models_dir + model)['model_state_dict'])
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
    while len(random_indices) < 100:
        random_indices.add(random.randint(0, len(f['images']) - 1))

    model.eval()
    with torch.no_grad():
        for i, idx in enumerate(list(random_indices)):
            print(f'{i + 1} / 100')
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