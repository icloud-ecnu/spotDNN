import numpy as np
from profiler import instanceinfo
import predictor
instance_type = instanceinfo.instance_type
instance_quota = instanceinfo.instance_quota
instance_time = instanceinfo.instance_time

def searchclustser_spotdnn(losstarget, timetarget):
    targetcuster = []
    mincost =  999999999
    targetimages  = 9999999

    instance_time_list = []
    index = 0
    for time in instance_time:
        item = [index, time]
        instance_time_list.append(item)
        index= index + 1
    instance_time_list.sort(key=lambda x: x[1], reverse=False)

    instance_list = []
    for item in instance_time_list:
        index = item[0]
        instance_list = instance_list + instance_quota[index]*[index]

    last_cluster = []
    for n in range(1,len(instance_list)):
        for m in range(len(instance_list)):
            if(m+n > len(instance_list)):
                cluster = instance_list[m:m+n] + instance_list[0:(m+n)%len(instance_list)]
            else: 
                cluster = instance_list[m:m+n]
            if(cluster == last_cluster): 
                continue
            else:
                last_cluster = cluster
            norm_batch, clus_speed = predictor.batch_and_speed(cluster)
            it = np.around(predictor.getit(losstarget, norm_batch, len(cluster)), decimals = 0)
            time = np.around(norm_batch * it / clus_speed, decimals = 2) 
            cost = predictor.computecost(cluster, time)
            if 0 < time <= timetarget and cost < mincost:
                mincost = cost
                targetcuster = cluster
                targetimages = it * norm_batch

    targetcuster_count = len(instance_type)*[0]
    for item in targetcuster:
        targetcuster_count[item] += 1

    return targetcuster_count, targetimages

