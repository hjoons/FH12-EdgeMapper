import socket
from typing import Any
import paramiko
import utils
import threading
import os
import argparse

# placeholders, get this from params
learning_rate = 0.01
num_epochs = 50
num_loops = 5

# p ="C:\\Users\\vliew\\Documents\\UTAustin\\Fall2023\ECE 364D - Senior Design\\FH12-EdgeMapper\\comm-test"
p = "~/ECE 364D - Senior Design/FH12-EdgeMapper/comm-test"

class DeviceHandler(threading.Thread):
    def __init__(self, device_ip: str, device_port: int, device_path: str, local_path: str, global_path: str):
        threading.Thread.__init__(self)
        self.device_ip = device_ip
        self.device_port = device_port
        self.device_path = device_path  # path to send the global models
        self.local_path = local_path    # path to receive federated models
        self.global_path = global_path  # path of the global model

    def run(self):
        # connect to device
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((self.device_ip, self.device_port))

        # connect message
        print('Sending connect message')
        utils.send_message(sock, "Connect")
        while True:
            msg = utils.receive_message(sock)
            if msg == "ACK":
                break
        print("Connect acknowledged")

        # setup message
        # learning rate, num_epochs, device local_path, device remote_path
        print("Sending setup message")
        msg = ";".join([str(learning_rate), str(num_epochs), self.device_path, self.local_path])
        utils.send_message(sock, msg)
        while True:
            msg = utils.receive_message(sock)
            if msg == "ACK":
                break
        print("Setup acknowledged")

        # send global model
        print("Sending global model")
        utils.send_scp_file(self.global_path, self.device_path, self.device_ip, self.device_port, "test_user", "pass", "fake_model.py", sock)
        print("Global model sent")

        while True:
            msg = utils.receive_message(sock)
            if msg == "Start":
                print("Device training started")
                break
        
        while True:
            msg = utils.receive_message(sock)
            if msg == "Done":
                print("Device training done")
                utils.send_message(sock, "ACK")
                break

        print("Waiting for device model")
        utils.receive_scp_file(self.local_path, sock)
        print("Device model received")

        sock.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Argument Parser for learning rate, num epochs, and num loops')

    parser.add_argument('--learning_rate', '--lr', type=float, default=0.01, help='Learning rate for the model')
    parser.add_argument('--num_epochs', '--ne', type=int, default=50, help='Number of epochs for the model')
    parser.add_argument('--num_loops', '--nl', type=int, default=5, help='Number of loops for the model')

    args = parser.parse_args()

    learning_rate = args.learning_rate
    num_epochs = args.num_epochs
    num_loops = args.num_loops

    
    for i in range(num_loops):
        # create device handlers
        dev1 = DeviceHandler('127.0.0.1', 8888, 'fake_model.py', 'fake_model.py', 'fake_model.py')
        
        # start device handler threads
        dev1.start()

        # wait for all threads to finish
        dev1.join()

        # aggregate models
        pass

