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
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((host, port))
    server_socket.listen(5)
    print('Server listening....')
    conn, addr = server_socket.accept()
    print('Connection from:', addr)
    # ssh_client = paramiko.SSHClient()
    # ssh_client.load_system_host_keys()
    # ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    # ssh_client.connect('localhost', username='your_username', password='your_password')
    # sftp = ssh_client.open_sftp()
    while True:
        data = receive_message(conn)
        print('Received message:', data)
        send_message(conn, 'Message received!')
        # local_path = 'local_file.zip'
        # remote_path = '/path/on/server/remote_file.zip'
        # receive_zip_file(sftp, remote_path, local_path)
    conn.close()
if __name__ == '__main__':
    main()
