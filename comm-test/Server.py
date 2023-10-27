import paramiko
from paramiko import SSHClient
from paramiko import Transport
from scp import SCPClient
import utils
import socket

# # Create an SSH client
# ssh = SSHClient()
# ssh.load_system_host_keys()
# ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# # Connect to the SSH server
# ssh.connect('127.0.0.1', port=8888, username='username', password='password')

# # Send a file to the server
# send_file(ssh, 'path_to_your_file.txt')

# # Close the SSH connection
# ssh.close()

server_host = '0.0.0.0'  # Listen on all available network interfaces
server_port = 2222  # Specify the SSH port

def main():
    while True:
        # Create SSH Server ???
        # TODO: idk how to do this
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((server_host, server_port))
        server_socket.listen(5)

        print(f"SSH server is listening on {server_host}:{server_port}") 
        # TODO: idk if there should be more LOL

        # connect
        msg = ''
        while msg != 'Connect':
            msg = utils.receive_message(channel)
        print('Received connect message')
        utils.send_message(channel, 'ACK')

        # Receive setup info
        msg = utils.receive_message(channel)
        split_msg = msg.split(';')
        learning_rate = split_msg[0]
        num_epochs = split_msg[1]
        local_path = split_msg[2]
        remote_path = split_msg[3]
        # other setup info??
        print('Received setup info')

        # After setup info receive global model
        utils.receive_scp_file(channel, local_path)
        print('Received global model')
        
        # Start Training ack message
        utils.send_message('Start')

        # train func
        print('Starting training')
        utils.train(learning_rate, num_epochs)
        print('Training complete')

        # training done
        utils.send_message('Done')
        while msg != 'ACK':
            msg = utils.receive_message(channel)
        
        # send model
        print('Sending local model')
        # utils.send_scp_file(stuff)
        print('Local model sent')

        # loop finished, close
        ssh.close() #TODO: do we have to do this if the client ends connection?

if __name__ == '__main__':
    main()