import paramiko
from paramiko import SSHClient
from paramiko import Transport
from paramiko.pkey import PKey
from scp import SCPClient
import utils
import socket
import threading

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

class SSHServer(paramiko.ServerInterface):
    def __init__(self):
        self.event = threading.Event()
        # self.add_server_key(paramiko.RSAKey.generate(2048))

    def set_server(self, server):
        self.server = server

    def check_auth_none(self, username: str) -> int:
        return 0
    
    def check_auth_password(self, username: str, password: str) -> int:
        return 0
    
    def check_auth_publickey(self, username: str, key: PKey) -> int:
        return 0

server_host = '0.0.0.0'  # Listen on all available network interfaces
server_port = 8888  # Specify the SSH port

def main():
    while True:
        # Create SSH Server ???
        # TODO: idk how to do this
        server = SSHServer()

        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((server_host, server_port))
        server_socket.listen(5)

        print(f"SSH server is listening on {server_host}:{server_port}") 

        server.set_server(server_socket)

        # TODO: idk if there should be more LOL

        # Accept a client connection
        client_socket, client_address = server_socket.accept()

        # Initialize the SSH server and transport
        transport = paramiko.Transport(client_socket)
        
        transport.add_server_key(paramiko.RSAKey.generate(2048))
        
      

        transport.start_server(server=server)

        # Accept the channel for SCP file transfer
        channel = transport.accept()
        if channel is None:
            print("SSH negotiation failed.")
            client_socket.close()
            exit(1)

        print(f"SSH connection established with {client_address[0]}:{client_address[1]}")


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
        # ssh.close() #TODO: do we have to do this if the client ends connection? --> i think server will always close connection?
        channel.close()
        transport.close()

if __name__ == '__main__':
    main()