import socket
import paramiko
import utils


#def send_message(conn, message):
#    conn.send(message.encode())
#def receive_message(conn):
#    return conn.recv(1024).decode()
#def send_zip_file(sftp, local_path, remote_path):
#    sftp.put(local_path, remote_path)
#def receive_zip_file(sftp, remote_path, local_path):
#    sftp.get(remote_path, local_path)


#def main():
#    host = '127.0.0.1'
#    port = 8888
#    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#    client_socket.connect((host, port))
#    while True:
#        message = input("Enter message: ")
#        send_message(client_socket, message)
#        data = receive_message(client_socket)
#        print('Received:', data)
#        # ssh_client = paramiko.SSHClient()
#        # ssh_client.load_system_host_keys()
#        # ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
#        # ssh_client.connect('localhost', username='your_username', password='your_password')
#        # sftp = ssh_client.open_sftp()
#        # local_path = 'local_file.zip'
#        # remote_path = '/path/on/server/remote_file.zip'
#        # send_zip_file(sftp, local_path, remote_path)
#    client_socket.close()
class SSHMessageClient:
    def __init__(self, hostname, port, username, password):
        self.hostname = hostname
        self.port = port
        self.username = username
        self.password = password
        self.ssh_client = paramiko.SSHClient()
        self.ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def connect(self):
        try:
            self.ssh_client.connect(self.hostname, self.port, self.username, self.password)
            print(f"Connected to {self.hostname} via SSH")
        except Exception as e:
            print(f"Error: {str(e)}")

    def send_message(self, message):
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command(f'echo "{message}" > message.txt')
            print(f"Sent message: {message}")
        except Exception as e:
            print(f"Error: {str(e)}")

    def receive_message(self):
        try:
            stdin, stdout, stderr = self.ssh_client.exec_command('cat message.txt')
            messages = stdout.read().decode()
            print(f"Received messages: {messages}")
            return messages
        except Exception as e:
            print(f"Error: {str(e)}. Resend")
            return None

    def disconnect(self):
        self.ssh_client.close()
        print("SSH connection closed")

def device_handler(self):
        status = True
        while status:
            msg = ''
            while msg != 'ACK':
                msg = self.receive_message()
            self.send_message("learning_rate;num_epochs;local_path;remote_path")
            utils.send_scp_file(self, local_path_to_model) # send the model
            self.send_message("filename;filesize")
            #maybe add something about sending if not received, work with server
            while msg != 'Start':
                msg = self.receive_message()
            self.send_message("ACK")
            while msg != 'Done':
                msg = self.receive_message()
            self.send_message("ACK")
            utils.receive_scp_file() # get updated model, figure params later
            print('Received local model')


if __name__ == "__main__":
    hostname = '127.0.0.1'
    port = 8888
    username = 'username'
    password = 'password'

    ssh_message_client = SSHMessageClient(hostname, port, username, password)
    ssh_message_client.connect()
    ssh_message_client.device_handler()


    
    # Send a message
    message_to_send = "Connect"
    ssh_message_client.send_message(message_to_send)

    # Receive messages
    received_messages = ssh_message_client.receive_message()
    
    ssh_message_client.disconnect()

