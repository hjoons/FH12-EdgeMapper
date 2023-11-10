import paramiko
import socket
from scp import SCPClient
import zipfile
import os
import time

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
      

def train(learning_rate, num_epochs):
    # not using any model information for this step
    for i in range(num_epochs):
        print(f"\tEpoch{i}")
        # adding 1 second time delay
        time.sleep(1)

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

