import socket
import paramiko
def send_message(conn, message):
    conn.send(message.encode())
def receive_message(conn):
    return conn.recv(1024).decode()
def send_zip_file(sftp, local_path, remote_path):
    sftp.put(local_path, remote_path)
def receive_zip_file(sftp, remote_path, local_path):
    sftp.get(remote_path, local_path)
def main():
    host = '127.0.0.1'
    port = 8888
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((host, port))
    while True:
        message = input("Enter message: ")
        send_message(client_socket, message)
        data = receive_message(client_socket)
        print('Received:', data)
        # ssh_client = paramiko.SSHClient()
        # ssh_client.load_system_host_keys()
        # ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        # ssh_client.connect('localhost', username='your_username', password='your_password')
        # sftp = ssh_client.open_sftp()
        # local_path = 'local_file.zip'
        # remote_path = '/path/on/server/remote_file.zip'
        # send_zip_file(sftp, local_path, remote_path)
    client_socket.close()
if __name__ == '__main__':
    main()
