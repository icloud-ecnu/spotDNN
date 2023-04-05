import numpy as np
import subprocess
from profiler import instanceinfo

def run(modelfile, datafile, logfile, targetimages):
    instance_gpus = instanceinfo.instance_gpus
    key = instanceinfo.key
    with open("launcher/instancesInfo.txt", "r") as f:
        lines = f.readlines()
        hosts = lines[0].split(' ')[:-1]
        hosts_private = lines[1].split(' ')[:-1]
        batchsize = lines[2].split(' ')[:-1]
        instance_index = lines[3].split(' ')[:-1]
    
    basebatch=256
    lr_adjust = []
    for b in batchsize:
        lr_adjust.append(int(b) // basebatch)
    
    ps_host = hosts[0] + ':5555'
    worker_host = ''
    port = 5556
    tag = 1
    for i in range(1, len(hosts_private)):
        for j in range(instance_gpus[int(instance_index[i])]):
            if(tag == 1):
                worker_host = worker_host + hosts_private[i] + ':' + str(port)
                port += 1
                tag = 0
            else:
                worker_host = worker_host + ',' + hosts_private[i] + ':' + str(port)
                port += 1
    
    command1 = '''source activate tensorflow_p37 && python %s \
                        --TF_FORCE_GPU_ALLOW_GROWTH=true \
                        --dataset=cifar100 \
                        --resnet_size=110 \
                        --batch_size=256 \
                        --lr_adjust=1 \
                        --max_imgs=%s \
                        --job_name=ps \
                        --task_index=0 \
                        --ps_hosts=%s \
                        --worker_hosts=%s \
                        --data_dir=%s \
                        --train_dir=%s \
                        --num_gpus=0 ''' %(modelfile, targetimages, ps_host, worker_host, datafile, logfile)

    ps_command = '''sleep 15 && ssh -o StrictHostKeyChecking=no -i %s.pem ubuntu@%s %s &''' %(key, hosts[0], command1)
    subprocess.run(ps_command, shell=True).decode()

    worker_index = 0
    for host_index in range(1, len(hosts)):
        for j in range(instance_gpus[int(instance_index[host_index])]):
            command1 = '''source activate tensorflow_p37 && python %s \
                                --TF_FORCE_GPU_ALLOW_GROWTH=true \
                                --dataset=cifar100 \
                                --resnet_size=110 \
                                --batch_size=%s \
                                --lr_adjust=%s \
                                --max_imgs=%s \
                                --job_name=worker \
                                --task_index=%s \
                                --ps_hosts=%s \
                                --worker_hosts=%s \
                                --num_gpus=%s \
                                --data_dir=%s \
                                --train_dir=%s ''' \
                                %(modelfile, batchsize[host_index], lr_adjust[host_index], \
                                targetimages, worker_index, ps_host, worker_host, \
                                instance_gpus[int(instance_index[host_index])],\
                                datafile, logfile,)
            worker_index += 1
            worker_command = '''sleep 15 && ssh -o StrictHostKeyChecking=no -i %s.pem ubuntu@%s %s &''' %(key, hosts[host_index], command1)
            subprocess.run(worker_command, shell=True).decode()
    
