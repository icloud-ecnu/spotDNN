import os
from statistics import mean
import pandas as pd
import numpy as np
from launcher import launcher
from profiler import instanceinfo
import run

def mergeAndSort(dirpath, metrics):
    fulldirpath = dirpath
    mergefilename = dirpath + "merge.txt"
    if os.path.isdir(fulldirpath):
        fileList = os.listdir(fulldirpath)
        mergefile = open(mergefilename, "w")
        for f in fileList:
            if f.startswith("worker"):
                workerfile = open(fulldirpath+f, 'r')
                lines = workerfile.readlines()
                for l in lines[3:]:
                    if l.startswith('2022'):
                        mergefile.write(l)
                workerfile.close()
        mergefile.close()
    if metrics == 0:
        data = pd.read_table(mergefilename, header=None, \
            names=['datetime', 'g_step', 'g_img', 'loss_value', 'examples_per_sec', 'sec_per_batch'])
        data = data.sort_values(by=['g_img'])
    else:
        data = pd.read_table(mergefilename, header=None, \
            names=['datetime', 'g_step', 'loss_value', 'examples_per_sec', 'sec_per_batch'])
        data = data.sort_values(by=['g_step'])
    return data

def CVandAvgbatch(dirpath, metrics):
    sortby = ['datetime', 'g_img'] if metrics == 0 else ['datetime', 'g_step']
    fulldirpath = dirpath
    if os.path.isdir(fulldirpath):
        fileList = os.listdir(fulldirpath)
        iterationTime = []
        speed = []
        batchSize = []
        for f in fileList:
            if f.startswith("worker"):
                workerfile = open(fulldirpath+f, 'r')
                data = pd.read_table(workerfile).sort_values(by=sortby)
                timeavg = data['sec_per_batch'][10:-3].mean()
                batch = int(f.split("_")[2].split("b")[1])
                iterationTime.append(timeavg)
                speed.append(batch/timeavg)
                batchSize.append(batch)
        iterationTimetrans = 1 / np.array(iterationTime)

        normbatch = np.around(np.dot(np.array(batchSize), iterationTimetrans) / np.sum(iterationTimetrans), decimals=1)
        n = len(batchSize)

    return normbatch, n

from scipy.optimize import curve_fit

def loss(X, a, b, c, d):  #幂函数
    x, avgbatch, n = X
    return  (((b * avgbatch + c) * np.sqrt(n)) / (x + a)) + d

def fitting_r(dirlist):
    fit_x = []
    fit_y = []
    fit_n = []
    fit_batch = []
    for dir in dirlist:
        data = mergeAndSort(dir, 0) 
        normbatch, n = CVandAvgbatch(dir, 0)
        x = data["g_img"] / normbatch 
        y = data['loss_value']
        fit_x += x
        fit_y += y
        fit_n += [n] * len(x)
        fit_batch += [normbatch] * len(x)
    fitted, _ = curve_fit(loss, (np.array(fit_x), np.array(fit_batch), np.array(fit_n)), fit_y)
    print('fitted:', fitted)
    return fitted

def fitting_bandwidth(dirlist):
    fulldirpath =  dirlist[0]
    if os.path.isdir(fulldirpath):
        fileList = os.listdir(fulldirpath)
        for f in fileList:
            if f.endswith("host0"):
                bandwith = []
                effband = []
                workerfile = open(fulldirpath+f, 'r')
                lines = workerfile.readlines()
                for l in lines[1:]:
                    b = int(l.split(',')[1]) / 100000
                    bandwith.append(b)
                for item in bandwith:
                    if item > 10:
                        effband.append(item)
                bandwidth = np.mean(effband)
                print(bandwidth)
    return bandwidth
    

    