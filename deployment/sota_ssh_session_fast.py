"""Interactive SSH/SFTP session. Credentials are read without echo, never saved."""
import getpass
import json
import pathlib
import sys
import paramiko

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('223.109.239.36', port=10316, username='root',
               password=getpass.getpass('SSH password: '),
               look_for_keys=False, allow_agent=False, timeout=20)
client.get_transport().set_keepalive(30)
sftp = paramiko.SFTPClient.from_transport(client.get_transport(), window_size=64*1024*1024)
print('SSH_READY', flush=True)
for line in sys.stdin:
    try:
        req = json.loads(line)
        op = req['op']
        if op == 'exec':
            _, stdout, stderr = client.exec_command(req['command'], timeout=req.get('timeout', 60))
            out = stdout.read().decode('utf-8', errors='replace')
            err = stderr.read().decode('utf-8', errors='replace')
            print(json.dumps({'stdout': out, 'stderr': err, 'exit': stdout.channel.recv_exit_status()}, ensure_ascii=False), flush=True)
        elif op == 'put':
            for src, dst in req['files']:
                sftp.put(src, dst)
            print(json.dumps({'uploaded': len(req['files'])}), flush=True)
        elif op == 'get':
            for src, dst in req['files']:
                pathlib.Path(dst).parent.mkdir(parents=True, exist_ok=True)
                sftp.get(src, dst)
            print(json.dumps({'downloaded': len(req['files'])}), flush=True)
        elif op == 'list':
            print(json.dumps([{'name': x.filename, 'size': x.st_size, 'mode': x.st_mode} for x in sftp.listdir_attr(req['path'])]), flush=True)
        elif op == 'close':
            break
        else:
            raise ValueError('Unknown operation')
    except Exception as exc:
        print(json.dumps({'error': str(exc)}), flush=True)
sftp.close()
client.close()
