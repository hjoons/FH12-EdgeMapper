import utils
import socket
import os
import torch

server_port = 8888

def main():

    server_host = '192.168.1.43'

    while True:
        # create server
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((server_host, server_port))
        server_socket.listen(5)

        print(f"Device is listening on {server_host}:{server_port}")

        # Accept a client connection
        sock, client_address = server_socket.accept()

        print(f"Cloud accepted from address: {client_address[0]}")

        # connect message
        while True:
            msg = utils.receive_message(sock)
            if msg == "Connect":
                utils.send_message(sock, "ACK")
                break

        # Receive setup info
        msg = utils.receive_message(sock)
        split_msg = msg.split(';')
        learning_rate = float(split_msg[0])
        num_epochs = int(split_msg[1])
        remote_path = split_msg[2]  # path to send the federated model
        dev_path = os.path.expanduser(split_msg[3])
        device_num = split_msg[4]
        cloud_user = split_msg[5]
        cloud_pwd = split_msg[6]
        
        if not os.path.exists(dev_path):
            os.makedirs(dev_path)
        os.chdir(dev_path)
        dev_path = ''
        print('Received setup info')
        utils.send_message(sock, "ACK")

        # After setup info receive global model
        print('Waiting for global model...')
        global_file = utils.receive_scp_file(dev_path, sock)
        print(f'Received global model: {global_file}')
        print(f"Model loaded!")
        
        # Start Training ack message
        utils.send_message(sock, 'Start')

        # train func
        print('Starting training')
        federated_file = f'federated_{device_num}.pt'
        federated_file = utils.train(learning_rate, num_epochs, global_file, federated_file)
        print('Training complete')
        
        print(f"Training saved as {dev_path + federated_file}")

        # training done
        utils.send_message(sock, 'Done')
        while msg != 'ACK':
            msg = utils.receive_message(sock)
        
        # send model
        federated_zip = f"federated_{device_num}.zip"
        print('Sending local model')
        utils.zip_file(dev_path + federated_file, dev_path + federated_zip)
        utils.send_scp_file(dev_path, remote_path, client_address[0], cloud_user, cloud_pwd, federated_file, sock, federated_zip)
        print('Local model sent')

        # delete local model and zip
        os.remove(dev_path + federated_file)
        os.remove(dev_path + federated_zip)
        os.remove(global_file)

        # loop finished, close
        sock.close()
        server_socket.close()

if __name__ == '__main__':
    main()