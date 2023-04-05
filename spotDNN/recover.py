# get all loss logs and fit the loss function -> loss objectives
# get time for now -> remaining time
import numpy as np
import argparse
import subprocess
import provisioner
from profiler import instanceinfo
import datetime

parser = argparse.ArgumentParser()
parser.add_argument('--host_index', type=int)
args = parser.parse_args()

host_index = args.host_index

remainingtime = (instanceinfo.objtimestamp - datetime.datetime.now()).total_seconds()

revHost = 0
with open("launcher/instancesInfo.txt", "r") as f:
    lines = f.readlines()
    hosts = lines[0].split(" ")[:-1]
    for host in hosts:
        print(host)
        cmd = '''scp -r -i %s ubuntu@%s:/home/ubuntu/log/* /home/ubuntu/log''' % (instanceinfo.key, host)
        subprocess.run(cmd, shell=True)


