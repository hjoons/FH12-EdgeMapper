import paramiko
import socket
from scp import SCPClient
import zipfile
import os
import time
import h5dataset
import torch
from frame_generator import UNet
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
    model = UNet().to(torch.device(device))
    model.load_state_dict(torch.load(f'{file_path}')['model_state_dict'])
    model.eval()
    # not using any model information for this step
        # print(f"\tEpoch{i}")
        # # adding 1 second time delay
        # time.sleep(1)
    train_loader, val_loader = h5dataset.createH5TrainLoader(path='/home/orin/Documents/FH12_23-24/FH12-EdgeMapper/comm-deploy/ecj1204.h5', batch_size=1)

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

