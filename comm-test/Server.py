import paramiko
from paramiko import SSHClient
from paramiko import Transport
from paramiko.pkey import PKey
from scp import SCPClient
import utils
import socket
import threading

server_host = '127.0.0.1'  # Listen on all available network interfaces
server_port = 8888  # Specify the SSH port

def main():
    while True:
        # create server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((server_host, server_port))
        server_socket.listen(5)

        print(f"Server is listening on {server_host}:{server_port}")

        # Accept a client connection
        sock, client_address = server_socket.accept()

        print(f"Client accepted from address: {client_address[0]}")

        # connect message
        while True:
            msg = utils.receive_message(sock)
            if msg == "Connect":
                utils.send_message(sock, "ACK")
                break

        # Receive setup info
        msg = utils.receive_message(sock)
        split_msg = msg.split(';')
        learning_rate = split_msg[0]
        num_epochs = split_msg[1]
        local_path = split_msg[2]
        remote_path = split_msg[3]
        # other setup info??
        print('Received setup info')
        utils.send_message(sock, "ACK")

        # After setup info receive global model
        utils.receive_scp_file(local_path, sock)
        print('Received global model')
        
        # Start Training ack message
        utils.send_message(sock, 'Start')

        # train func
        print('Starting training')
        utils.train(learning_rate, num_epochs)
        print('Training complete')

        # training done
        utils.send_message(sock, 'Done')
        while msg != 'ACK':
            msg = utils.receive_message(sock)
        
        # send model
        print('Sending local model')
        utils.send_scp_file(local_path, remote_path, client_address[0], client_address[1], "test_user", "pass", "fake_model.py", sock)
        print('Local model sent')

        # loop finished, close
        sock.close()
        server_socket.close()

if __name__ == '__main__':
    main()