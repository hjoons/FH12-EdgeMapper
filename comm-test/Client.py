import paramiko

ssh = paramiko.SSHClient()

ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

ssh.connect('127.0.0.1', port=8888, username='username', password='password')
try:
    stdin, stdout, stderr = ssh.exec_command('ls -l')
    print(stdout.read().decode('utf-8'))
except paramiko.SSHException as e:
    print(f'ssh exception: {e}')
