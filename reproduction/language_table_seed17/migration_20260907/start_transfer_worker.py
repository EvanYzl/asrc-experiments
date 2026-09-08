"""Send an SSH password directly to an isolated worker's stdin, never to a file."""
from pathlib import Path
import getpass
import subprocess
import sys

base=Path(__file__).resolve().parent
password=getpass.getpass('SSH/SFTP authentication password: ')
with (base/'transfer_worker.log').open('ab') as output:
    child=subprocess.Popen([sys.executable,'-B',str(base/'transfer_worker.py')],cwd=base,stdin=subprocess.PIPE,
        stdout=output,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
    child.stdin.write((password+'\n').encode('utf-8'));child.stdin.flush();child.stdin.close()
password=None
print('Background transfer worker PID:',child.pid,flush=True)
