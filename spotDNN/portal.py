import numpy as np
import subprocess
import provisioner
from profiler import instanceinfo
from launcher import launcher
import run
import time
import datetime
# performance profiler
    # 1. launch a group of instances and then run a few iterations (launcher.py + run.py)
    # 2. get training logs including speed, loss, parameter size and so on listed in instanceinfo.py (profiler.py)

# provisioner (include predictor)
objloss = 0.8
objtime = 2400
instance_count, targetimages = provisioner.searchclustser_spotdnn(objloss, objtime)

# launcher
instance_type = instanceinfo.instance_type
hosts = launcher.getSpotInstance(instance_type, instance_count)

instanceinfo.starttimestamp = datetime.datetime.now()
instanceinfo.objtimestamp = instanceinfo.starttimestamp + objtime

run.run(instanceinfo.modelfile, instanceinfo.datafile, instanceinfo.logfile, int(targetimages))

# detector
detector_cmd = """sh detector.sh"""
subprocess.run(detector_cmd, shell=True).decode()