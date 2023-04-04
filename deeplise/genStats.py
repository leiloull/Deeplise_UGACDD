from argparse import ArgumentParser
import json
from math import sqrt, floor, ceil
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import numpy as np

def accuracy(tp, fp, tn, fn):
    acc = (tp+tn)/(tp+fp+tn+fn)
    return acc

def sensitivity(tp, fp, tn, fn):
    try:
        sense = tp/(tp+fn)
    except:
        sense = 1.0
    return sense

def specificity(tp, fp, tn, fn):
    try:
        spec = tn/(tn+fp)
    except:
        spec = 1.0
    return spec

def precision(tp, fp, tn, fn):
    try:
        prec = tp/(tp+fp)
    except:
        prec = 1.0
    return prec

def mcc(tp, fp, tn, fn):
    N = tp+fp+tn+fn
    S = (tp+fn)/N
    P = (tp+fp)/N
    if S == 0:
        S = 0.0001
    if P == 0:
        P = 0.0001
    MCC = ((tp/N)-(S*P))/sqrt(P*S*(1.0-S)*(1.0-P))
    return MCC

def genStatProc(tasks_to_accomplish, tasks_that_are_done, jsonDir):

    tpList = []
    fpList = []
    tnList = []
    fnList = []
    avgStatList = []

    while True:
        try:
            '''
                try to get task from the queue. get_nowait() function will 
                raise queue.Empty exception if the queue is empty. 
                queue(False) function would do the same task also.
            '''
            filename = tasks_to_accomplish.get_nowait()
            print("Processing: " + filename + ", Remaining: " + str(tasks_to_accomplish.qsize()))
        except queue.Empty:

            break
        else:
            '''
                if no exception has been raised, add the task completion 
                message to task_that_are_done queue
            '''
            with open(os.path.join(jsonDir, filename), 'r') as inputFile:

                currentJson = json.loads(inputFile.read())
                tp = (currentJson['Stats']['True Positive'])
                fp = (currentJson['Stats']['False Positive'])
                tn = (currentJson['Stats']['True Negative'])
                fn = (currentJson['Stats']['False Negative'])

                tpList.append(tp)
                fpList.append(fp)
                tnList.append(tn)
                fnList.append(fn)
                
                currentStats = []
                currentStats.append(accuracy(tp, fp, tn, fn))
                currentStats.append(sensitivity(tp, fp, tn, fn))
                currentStats.append(specificity(tp, fp, tn, fn))
                currentStats.append(precision(tp, fp, tn, fn))
                currentStats.append(mcc(tp, fp, tn, fn))

                avgStatList.append(currentStats)
                
                
                 
            tasks_that_are_done.put(filename + ' is done by ' + current_process().name)
            # time.sleep(.5)
    
    print('Global Stats')
    globalTp = sum(tpList)
    globalFp = sum(fpList)
    globalTn = sum(tnList)
    globalFn = sum(fnList)

    # for i in range(0, len(tpList)):
    #     globalTp += tpList[i]
    #     globalFp += fpList[i]
    #     globalTn += tnList[i]
    #     globalFn += fnList[i]

    print('Global Accuracy: ' + str(accuracy(globalTp, globalFp, globalTn, globalFn)))
    print('Global Sensitivity,: ' + str(sensitivity(globalTp, globalFp, globalTn, globalFn)))
    print('Global Specificity: ' + str(specificity(globalTp, globalFp, globalTn, globalFn)))
    print('Global Precision: ' + str(precision(globalTp, globalFp, globalTn, globalFn)))
    print('Global MCC: ' + str(mcc(globalTp, globalFp, globalTn, globalFn)))

    avgStats = [0.0, 0.0, 0.0, 0.0, 0.0]

    for i in range(0, len(avgStatList)):
        for j in range(0, len(avgStats)):
            avgStats[j] += avgStatList[i][j]

    for j in range(0, len(avgStats)):
        avgStats[j] = avgStats[j] / len(avgStatList)

    print('Average Accuracy: ' + str(avgStats[0]))
    print('Average Sensitivity,: ' + str(avgStats[1]))
    print('Average Specificity: ' + str(avgStats[2]))
    print('Average Precision: ' + str(avgStats[3]))
    print('Average MCC: ' + str(avgStats[4]))
    
    return True

def genStat(jsonDir, num_workers=1):
    number_of_processes = num_workers
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    # print("Training")
    for filename in os.listdir(jsonDir):
        tasks_to_accomplish.put(filename)

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=genStatProc, args=(tasks_to_accomplish, tasks_that_are_done, jsonDir))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    proteinNum = 0
    while not tasks_that_are_done.empty():
        # print("Protein Num: " + str(proteinNum))
        # print(tasks_that_are_done.get())
        proteinNum += 1

def main(hparams):
    jsonDir = "data/outputJsons/"

    genStat(jsonDir, num_workers=1)

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.00005, help="Ranger: learning rate")
    parser.add_argument("--image_scale", type=float, default=1.0, help="What size the images should be")
    parser.add_argument("--val_percent", type=float, default=0.1, help="What percentage of the dataset you should use to validate the model")
    parser.add_argument("--n_channels", type=int, default=19, help="Number of input channels. In this case, it should always be 3.")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of output channels. In this case, it should also be 3.")
    parser.add_argument("--bilinear", type=bool, default=False, help="Determines if the model should use a learned upscaling or not. By default it chooses to learn.")
    parser.add_argument("--num_tpu_cores", type=int, default=None, help="Number of TPU cores to use. I have this as a holdover for when I used this skeleton on Google TPUs.")

    hparams = parser.parse_args()

    main(hparams)
