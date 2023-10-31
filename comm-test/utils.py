import paramiko
import socket
from scp import SCPClient
import zipfile
import os
import time

def zip_file(file, zip_file_name):
    try:
        with zipfile.ZipFile(zip_file_name, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(file, arcname=file)
        print(f'{file} has been zipped to {zip_file_name}')
    except Exception as e:
        print(f'Error zipping file: {e}')

def unzip_file(zip_file, destination_folder):
    try:
        with zipfile.ZipFile(zip_file, 'r') as zipf:
            zipf.extractall(destination_folder)
        print(f'{zip_file} has been successfully unzipped to {destination_folder}')
    except Exception as e:
        print(f'Error unzipping file: {e}')
      

def train(learning_rate, num_epochs):
    # not using any model information for this step
    for i in range(0,5):
        print(i)
        # adding 1 second time delay
        time.sleep(1)

def send_message(sock: socket.socket, msg: str):
    sock.sendall(msg.encode())

def receive_message(sock: socket.socket) -> str:
    msg = sock.recv(1024)
    return msg.decode()

def send_scp_file(local_path: str, remote_path: str, ip: str, port: int, user: str, pwd: str, file_name: str, sock: socket.socket):
    # TODO: zip here
    zip_file(local_path, "fake_model.zip")
    # TODO: get file size here
    file_size = os.path.getsize(local_path)
    
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=ip, username='vliew', password='U8133051', banner_timeout=200)

    received = False
    while not received:
        try:
            with SCPClient(ssh.get_transport()) as scp:
                scp.put(local_path, remote_path)
                print(f"Sent file: {local_path}")
        except Exception as e:
            print(f"Error sending file: {e}")
            continue
        
        msg = remote_path + ";" + file_name + ";" + str(file_size)
        send_message(sock, msg)

        msg = receive_message(sock)
        if msg == 'Received':
            received = True

    # TODO: delete zipped file here

def receive_scp_file(destination_path: str, sock: socket.socket):
    received = False
    while not received:
        msg = receive_message(sock)
        split_msg = msg.split(';')

        if os.path.exists(split_msg[0]):
            # TODO: unzip
            unzip_file(split_msg[0], destination_path)

            file_size = os.path.getsize(destination_path)
            # if (file_size == split_msg[2]):
            #     send_message(sock, "Received")
            #     received = True
            # else:
            #     send_message(sock, "Resend")
            send_message(sock, "Received")

            # TODO: delete zip file

            received = True

