import socket
from typing import Any
import utils
import threading
import os
import argparse
import torch
import shutil

# placeholders, get this from params
learning_rate = 0.01
num_epochs = 50
num_loops = 5

dev_ip = ['192.168.1.147',
          '127.0.0.1',
          '127.0.0.1',
          '127.0.0.1']

dev_port = [8888,
            8888,
            8888,
            8888]

dev_users = ['orin',
            'vliew',
            'orin',
            'orin',]

dev_pwds = ['12345678',
            'U8133051',
            '12345678',
            '12345678',]

device_path = ['~/Documents/FH12_23-24/device_training/',
               'C:/Users/vliew/Documents/FH12_23-24/device_training',
               '~/Documents/FH12_23-24/device_training/',
               '~/Documents/FH12_23-24/device_training/',]

user = 'vliew'
pwd = 'U8133051'

global_path = ''

global_file_path = 'mbnv3_epoch_100.pt'


class DeviceHandler(threading.Thread):
    """
    Device Handler class for communications between device and cloud

    Threaded class to handle device-cloud communications. New object needed per communication round per device.

    Attributes:
        device_num  (int): Number of the device
        device_ip   (str): ipv4 address of the device
        device_port (int): device port number for socket messages
        cloud_path  (int): path to receive federated models from device to cloud
        dev_user    (str): device username for paramiko ssh
        dev_pwd     (str): device password for paramiko ssh

    Methods:
        __init__(self, device_num: int, device_ip: str, device_port: int, cloud_path: str, dev_user: str, dev_pwd: str): class constructor
        run(self): run function for thread execution of communication protocol
    """

    def __init__(self, device_num: int, device_ip: str, device_port: int, cloud_path: str, dev_user: str, dev_pwd: str):
        threading.Thread.__init__(self)
        self.device_num = device_num    # device number
        self.device_ip = device_ip      # device ipv4 address
        self.device_port = device_port  # device port number for socket messages
        self.cloud_path = cloud_path    # path to receive federated models from device to cloud
        self.dev_user = dev_user        # device ssh username
        self.dev_pwd = dev_pwd          # device ssh password

    def run(self):
        # connect to device
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.device_ip, self.device_port))

        # connect message
        print(f'Dev{self.device_num}: Sending connect message')
        utils.send_message(sock, "Connect")
        while True:
            msg = utils.receive_message(sock)
            if msg == "ACK":
                break
        print(f"Dev{self.device_num}: Connect acknowledged")

        # setup message
        # learning rate, num_epochs, device path, cloud path, device_num
        print(f"Dev{self.device_num}: Sending setup message")
        msg = ";".join([str(learning_rate), str(num_epochs), self.cloud_path, device_path[self.device_num], str(self.device_num), user, pwd])
        utils.send_message(sock, msg)
        while True:
            msg = utils.receive_message(sock)
            if msg == "ACK":
                break
        print(f"Dev{self.device_num}: Setup acknowledged")
        print()

        # send global model
        print(f"Dev{self.device_num}: Sending global model")
        utils.send_scp_file(global_path, device_path[self.device_num], self.device_ip, self.dev_user, self.dev_pwd, global_file_path, sock, "global_model.zip")
        print(f"Dev{self.device_num}: Global model sent")
        print()

        while True:
            msg = utils.receive_message(sock)
            if msg == "Start":
                print(f"Dev{self.device_num}: Training started")
                break
        
        while True:
            msg = utils.receive_message(sock)
            if msg == "Done":
                print(f"Dev{self.device_num}: Training done")
                utils.send_message(sock, "ACK")
                break
        print()
        print(f"Dev{self.device_num}: Waiting for device model")
        pth = utils.receive_scp_file(self.cloud_path, sock)
        print(f"Dev{self.device_num}: Device model received")

        sock.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser for learning rate, num epochs, and num loops')

    parser.add_argument('--learning_rate', '--lr', type=float, default=0.01, help='Learning rate for the model')
    parser.add_argument('--num_epochs', '--e', type=int, default=1, help='Number of epochs for the model')
    parser.add_argument('--comm_rounds', '--r', type=int, default=2, help='Number of communication rounds for the model')
    parser.add_argument('--num_devices', '--n', type=int, default=1, help='Number of devices')

    args = parser.parse_args()

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_loops = args.comm_rounds
    num_devices = args.num_devices

    p = os.getcwd().replace('\\', '/') + '/'

    if not os.path.exists('federated_models/'):
        os.makedirs('federated_models/')

    # global_file_path = 'epoch_250.pt'
    
    for i in range(num_loops):
        # check for global model
        # if not os.path.exists('global_model.pth'):
        #     # TODO: replace with actual model
        #     with open('global_model.pth', "w") as file:
        #         file.write("Fake global model to send to edge devices")

        # zip global model
        print('Zipping global model...')
        utils.zip_file(global_file_path, 'global_model.zip')
        print('Finished zipping\n')
        print()

        # create device handlers
        # parameters: (device_num: int, device_ip: str, device_port: int, device_path: str, cloud_path: str)
        devs = []
        for k in range(num_devices):
            dev = DeviceHandler(k, dev_ip[k], dev_port[k], p + 'federated_models/', dev_users[k], dev_pwds[k])
            devs.append(dev)
        
        # start device handler threads
        for dev in devs:
            dev.start()

        # wait for all threads to finish
        for dev in devs:
            dev.join()

        # aggregate models
        print('\nAggregating models...')
        # TODO: aggregate models
        fed_model = utils.federated_averaging('federated_models/')
        print('Models aggregated')
        print('Evaluating global model...\n')

        utils.federated_eval('C:/Users/vliew/Documents/UTAustin/Fall2023/SeniorDesign/FH12-EdgeMapper/Device1/orgspace.h5')

        # delete federated models after aggregation and global zip
        for k in range(num_devices):
            os.remove(f'federated_models/federated_{k}.pt')
        os.remove('global_model.zip')