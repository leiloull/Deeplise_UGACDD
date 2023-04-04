import pandas as pd
import numpy as np
import json
import os
from math import sqrt
from argparse import ArgumentParser
import json
from math import sqrt, floor, ceil
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle

from sklearn import svm, datasets, linear_model
from sklearn.metrics import roc_curve, auc, matthews_corrcoef, precision_score, accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import statistics

def prepareData(currentJson):
    pointCount = 0
    statList = []
    currentList = []

    for mol in currentJson['molecules']:
        if mol['mol_name'] == "New Points":
            for r in mol['residues']:
                for a in r['atoms']:
                    if (a['atom_type'] == 'N'):
                        pointCount += 1
                        currentList.append(a['relative_affinity'])
                        if (a['relative_affinity'] > 1) and normal:
                            print("Odd score: " + str(a['relative_affinity']) + " in " + filename)
                            print("Full Dump below:")
                            print(a)
                            normal = False
                
    mean = statistics.mean(currentList)
    std = statistics.stdev(currentList)
    maxValue = max(currentList)
    stats = [mean, std, maxValue]
    statList.append(stats)
    
    Xr = np.empty(pointCount, dtype=np.float32)
    Xn = np.empty(pointCount, dtype=np.float32)
    Xo = np.empty(pointCount, dtype=np.float32)
    Xs = np.empty(pointCount, dtype=np.float32)
    y = np.empty(pointCount, dtype=np.int8)

    index = 0
    for mol in currentJson['molecules']:
        if mol['mol_name'] == "New Points":
            for r in mol['residues']:
                for a in r['atoms']:
                    if (a['atom_type'] == 'N'):
                        # Standardized
                        Xs[index] = (a['relative_affinity'] - statList[0][0]) / statList[0][1]
                        # Normalized
                        Xn[index] = (a['relative_affinity'] / statList[0][2])
                        # Odd
                        Xo[index] = (a['relative_affinity'] / statList[0][0])
                        # Raw
                        Xr[index] = a['relative_affinity']
                        if a['true_positive']:
                            y[index] = 1
                        else:
                            y[index] = 0
                        index += 1

    Xr = Xr.reshape(-1, 1)
    Xn = Xn.reshape(-1, 1)
    Xo = Xo.reshape(-1, 1)
    Xs = Xs.reshape(-1, 1)
    y = label_binarize(y, classes=[0, 1])

    return [Xr, y, Xn, Xo, Xs]

def genROC(X, y):

    n_classes = y.shape[1]
    random_state = np.random.RandomState(1)

    X_train = X
    X_test = X
    y_train = y
    y_test = y
    # Learn to predict each class against the other
    classifier = linear_model.SGDClassifier(loss='hinge', n_jobs=12, max_iter=1000, random_state=random_state)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    currentAUC = 0
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:])
        roc_auc[i] = auc(fpr[i], tpr[i])
        p = classifier.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
        currentAUC = auc(fpr[i], tpr[i])

    return currentAUC

def makeClusterDF(jsonDir, valDir):
    proteinNumber = 0
    nameList = []
    aucList = []
    
    for filename in os.listdir(jsonDir):

        if proteinNumber % 10 == 0:
            print(proteinNumber)

        if '.json' not in filename:
            continue

        with open(os.path.join(jsonDir, filename), 'r') as jsonFile:
            proteinJson = json.load(jsonFile)
            nameList.append(proteinJson['identifier'])

            dataArrays = prepareData(proteinJson)
            try:
                aucList.append(genROC(dataArrays[0], dataArrays[1]))
            except:
                print(nameList[proteinNumber])
                aucList.append(-999999)

        proteinNumber += 1
    
    data = {'Name': nameList, 
            'AUC': aucList}
    
    df = pd.DataFrame.from_dict(data)

    return df

if __name__ == '__main__':

    jsonDir = 'data/outputJsons'
    valDir = 'data/val-grids'

    clusterDF = makeClusterDF(jsonDir, valDir)

    csv_data = clusterDF.to_csv('GfG-AUC.csv', index = True) 
    print('\nCSV String:\n', csv_data) 

    
