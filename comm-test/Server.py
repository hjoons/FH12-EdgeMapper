import paramiko
from paramiko import SSHClient
from paramiko import Transport
from scp import SCPClient

# Create an SSH client
ssh = SSHClient()
ssh.load_system_host_keys()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

# Connect to the SSH server
ssh.connect('127.0.0.1', port=8888, username='username', password='password')

# Send a file to the server
send_file(ssh, 'path_to_your_file.txt')

# Close the SSH connection
ssh.close()