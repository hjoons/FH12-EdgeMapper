import socket
import paramiko
import utils
import Server


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
        self.channel = paramiko.Channel(2)

    def connect(self):
        print('Trying to connect')
        try:
            self.ssh_client.connect(self.hostname, self.port, self.username, self.password)
            print(f"Connected to {self.hostname} via SSH")
            self.channel = self.ssh_client.get_transport().open_channel('x11')
        except Exception as e:
            print(f"Error: {str(e)}")
            exit(1)


    def disconnect(self):
        self.ssh_client.close()
        print("SSH connection closed")

    def device_handler(self):
            status = True
            while status:
                msg = ''
                while msg != 'ACK':
                    msg = utils.receive_message(self.channel)
                utils.send_message(self.channel, "learning_rate;num_epochs;local_path;remote_path")
                utils.send_scp_file(self, '/fake_model.py') # send the model
                msg = "filename;filesize"
                utils.send_message(self.channel, msg)
                #maybe add something about sending if not received, work with server
                while msg != 'Start':
                    msg = utils.receive_message(self.channel)
                utils.send_message(self.channel, "ACK")
                while msg != 'Done':
                    msg = utils.receive_message(self.channel)
                utils.send_message(self.channel, "ACK")
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
    utils.send_message(ssh_message_client.channel, message_to_send)

    # Receive messages
    received_messages = utils.receive_message(ssh_message_client.channel)
    
    ssh_message_client.disconnect()

