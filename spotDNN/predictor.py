import numpy as np
from profiler import instanceinfo
instance_quota = instanceinfo.instance_quota
instance_spot_price = instanceinfo.instance_spot_price
instance_batch = instanceinfo.instance_batch
instance_gpus = instanceinfo.instance_gpus

param = instanceinfo.param
instance_bandwith = instanceinfo.instance_bandwith
instance_comp = instanceinfo.instance_comp

a = instanceinfo.a
b = instanceinfo.b
bps = instanceinfo.bps
r1 = instanceinfo.r1
r2 = instanceinfo.r2
r3 = instanceinfo.r3
r4 = instanceinfo.r4

def p(n):
    return min (a * np.sqrt(n) - b, 1)

def batch_and_speed(cluster):
    n = len(cluster)
    if(n * instance_bandwith > bps):
        band_real = (p(n) * bps / n) + (1-p(n)) * instance_bandwith
    else:
        band_real = instance_bandwith

    batch = []
    time = []
    for i in cluster:
        batch.append(instance_batch[i]*instance_gpus[i])
        time.append(instance_comp[i] + 2 * param / band_real)

    speed = np.array(batch) / np.array(time)
    iterationTimetrans = 1 / np.array(time)
    norm_batch = np.around(np.dot(np.array(batch), iterationTimetrans) / np.sum(iterationTimetrans), decimals=1)
    clus_speed = np.sum(speed) 

    return norm_batch, clus_speed

def computecost(cluster, time):
    unitprice = 0
    for i in cluster:
        unitprice = unitprice + instance_spot_price[i]
    return np.around(unitprice * time / 3600, decimals=6)

def getit(loss, norm_batch, instance_number):
    return (((r2* norm_batch + r3) * np.sqrt(instance_number)) / (loss - r4)) - r1
