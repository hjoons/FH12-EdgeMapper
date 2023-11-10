import socket
from typing import Any
import utils
import threading
import os
import argparse

# placeholders, get this from params
learning_rate = 0.01
num_epochs = 50
num_loops = 5

dev_ip = ['127.0.0.1',
          '127.0.0.1',
          '127.0.0.1',
          '127.0.0.1']

dev_port = [8888, 8889, 8890, 8891]

num_devices = 4

global_path = ''

# p ="C:/Users/vliew/Documents/UTAustin/Fall2023/ECE 364D - Senior Design/FH12-EdgeMapper/comm-test"
# p = 'C:/Users/vliew/Documents/UTAustin/Fall2023/SeniorDesign/FH12-EdgeMapper/comm-test/'


class DeviceHandler(threading.Thread):
    def __init__(self, device_num: int, device_ip: str, device_port: int, cloud_path: str):
        threading.Thread.__init__(self)
        self.device_num = device_num
        self.device_ip = device_ip
        self.device_port = device_port
        # self.device_path = device_path  # path to send the global models on device
        self.cloud_path = cloud_path    # path to receive federated models from device to cloud

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
        msg = ";".join([str(learning_rate), str(num_epochs), self.cloud_path, str(self.device_num)])
        utils.send_message(sock, msg)
        while True:
            msg = utils.receive_message(sock)
            if msg == "ACK":
                break
        print(f"Dev{self.device_num}: Setup acknowledged")

        device_path = utils.receive_message(sock)
        print(f"Dev{self.device_num}: Device path received")

        # send global model
        print(f"Dev{self.device_num}: Sending global model")
        utils.send_scp_file(global_path, device_path, self.device_ip, "vliew", "U8133051", "global_model.pth", sock, "global_model.zip")
        print(f"Dev{self.device_num}: Global model sent")

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

        print(f"Dev{self.device_num}: Waiting for device model")
        pth = utils.receive_scp_file(self.cloud_path, sock)
        print(f"Dev{self.device_num}: Device model received")

        sock.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser for learning rate, num epochs, and num loops')

    parser.add_argument('--learning_rate', '--lr', type=float, default=0.01, help='Learning rate for the model')
    parser.add_argument('--epochs', '--e', type=int, default=50, help='Number of epochs for the model')
    parser.add_argument('--comm_rounds', '--r', type=int, default=5, help='Number of communication rounds for the model')

    args = parser.parse_args()

    learning_rate = args.learning_rate
    num_epochs = args.epochs
    num_loops = args.comm_rounds

    p = os.getcwd().replace('\\', '/') + '/'

    if not os.path.exists('federated_models/'):
        os.makedirs('federated_models/')
    
    for i in range(num_loops):
        # check for global model
        if not os.path.exists('global_model.pth'):
            # TODO: replace with actual model
            with open('global_model.pth', "w") as file:
                file.write("Fake global model to send to edge devices")

        # zip global model
        utils.zip_file('global_model.pth', 'global_model.zip')

        # create device handlers
        # parameters: (device_num: int, device_ip: str, device_port: int, cloud_path: str)
        devs = []
        for k in range(num_devices):
            dev = DeviceHandler(k, dev_ip[k], dev_port[k], p + 'federated_models/')
            devs.append(dev)
        
        # start device handler threads
        for dev in devs:
            dev.start()

        # wait for all threads to finish
        for dev in devs:
            dev.join()

        # aggregate models
        # TODO: aggregate models
        print('Models aggregated')

        # delete federated models after aggregation and global zip
        for k in range(num_devices):
            os.remove(f'federated_models/dev{k}/federated_{k}.pth')
        os.remove('global_model.zip')


