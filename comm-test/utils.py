import paramiko
from paramiko import SSHClient
from paramiko import Transport
from scp import SCPClient
import zipfile
import os
import time

# def send_message(channel, message: str):
#     channel.send(message + "\n")

# def receive_message(channel) -> str:
#     data = channel.recv(1024)
#     return data.decode("utf-8").strip()

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
      

def send_scp_file(client, channel, local_file, remote_path, file_name, file_size):
    zip_file(file_name, 'something.zip')
    received = False

    while not received:
        try:
            with SCPClient(client.get_transport()) as scp:
                scp.put(local_file, remote_path)
                print(f"Sent file: {local_file}")
        except Exception as e:
            print(f"Error sending file: {e}")

        msg = file_name + ';' + file_size
        send_message(channel, msg)

        msg = receive_message(channel)
        if msg == 'Received':
            received = True

    
def receive_scp_file(channel, destination_file_path: str):
    received = False
    while not received:
        msg = receive_message(channel)
        split_msg = msg.split(';')
        if os.path.exists(split_msg[0]):
            unzip_file(split_msg[0], destination_file_path)

            # check file size equal
            file_size = os.path.getsize(destination_file_path)
            if file_size == int(split_msg[1]):
                send_message(channel, "Received")
                received = True
            else:
                send_message(channel, "Resend")


def train(learning_rate, num_epochs):
    # not using any model information for this step
    for i in range(0,5):
        print(i)
        # adding 1 second time delay
        time.sleep(1)