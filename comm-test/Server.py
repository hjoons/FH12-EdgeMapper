import socket
import threading
import paramiko


class SSHServer(paramiko.ServerInterface):
    def __init__(self):
        self.event = threading.Event()

    def check_channel_request(self, kind, chanid):
        if kind == 'session':
            return paramiko.OPEN_SUCCEEDED
        return paramiko.OPEN_FAILED_ADMINISTRATIVELY_PROHIBITED

    def check_auth_password(self, username, password):
        if username == 'username' and password == 'password':
            return paramiko.AUTH_SUCCESSFUL
        return paramiko.AUTH_FAILED


server_host = '127.0.0.1'
server_port = 8888

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

server_socket.bind((server_host, server_port))

server_socket.listen()

print(f"Server is listening on {server_host}:{server_port}")

while True:
    # Accept a client connection
    client_socket, client_address = server_socket.accept()

    transport = paramiko.Transport(client_socket)
    transport.add_server_key(paramiko.RSAKey.generate(2048))
    server = SSHServer()

    transport.start_server(server=server)

    channel = transport.accept()
    if channel is None:
        print("SSH negotiation failed.")
        client_socket.close()
        continue

    print("SSH connection established.")

    channel.send("Welcome to the custom SSH server!")
