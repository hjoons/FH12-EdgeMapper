import utils
import socket
import argparse
import os

server_host = '127.0.0.1'  # Listen on all available network interfaces
server_port = 8888  # Specify the SSH port

dev_ip = ['127.0.0.1',
          '127.0.0.1',
          '127.0.0.1',
          '127.0.0.1']

dev_port = [8888, 8889, 8890, 8891]

def main():
    parser = argparse.ArgumentParser(description='Argument Parser for device number')

    parser.add_argument('--device_number', '--n', type=int, default=0, help='Device number')
    args = parser.parse_args()

    server_host = dev_ip[args.device_number]
    server_port = dev_port[args.device_number]

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
        device_num = split_msg[3]
        
        dev_path = f"dev{device_num}/"
        if not os.path.exists(dev_path):
            os.makedirs(dev_path)
        print('Received setup info')
        utils.send_message(sock, "ACK")

        # After setup info receive global model
        global_file = utils.receive_scp_file(dev_path, sock)
        print(f'Received global model: {global_file}')
        
        # Start Training ack message
        utils.send_message(sock, 'Start')

        # train func
        print('Starting training')
        utils.train(learning_rate, num_epochs)
        print('Training complete')

        # dummy trained file
        federated_file = f"federated_{device_num}.pth"
        with open(dev_path + federated_file, "w") as file:
            file.write("Dummy file for federated training")
        print(f"Training saved as {dev_path + federated_file}")

        # training done
        utils.send_message(sock, 'Done')
        while msg != 'ACK':
            msg = utils.receive_message(sock)
        
        # send model
        print('Sending local model')
        utils.send_scp_file(dev_path, remote_path, client_address[0], "vliew", "U8133051", federated_file, sock, f"federated_{device_num}.zip")
        print('Local model sent')

        # delete local model??
        os.remove(dev_path + federated_file)

        # loop finished, close
        sock.close()
        server_socket.close()

if __name__ == '__main__':
    main()