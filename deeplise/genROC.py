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

def prepareData(jsonDir):
    print("Determining Size of Arrays")

    pointCount = 0
    fileCount = 0
    statList = []
    validList = []
    with open('valid.csv', newline='') as csvfile:
        spamreader = csv.reader(csvfile)
        for row in spamreader:
            validList.append(row[0])

    for filename in os.listdir(jsonDir):
        normal = True
        with open(os.path.join(jsonDir, filename), 'r') as inputFile:
            currentList = []
            currentJson = json.loads(inputFile.read())
            if currentJson['identifier'] not in validList:
                print('Passing over ' + currentJson['identifier'])
                continue
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
        if fileCount % 10 == 0:
            print(str(fileCount) + ':' + str(statList[fileCount][0]) + ' ' + str(statList[fileCount][1]))
        fileCount += 1

    print('Total Points: ' + str(pointCount))
    print('Making Arrays')
    
    Xr = np.empty(pointCount, dtype=np.float32)
    Xn = np.empty(pointCount, dtype=np.float32)
    Xo = np.empty(pointCount, dtype=np.float32)
    Xs = np.empty(pointCount, dtype=np.float32)
    y = np.empty(pointCount, dtype=np.int8)

    index = 0
    fileCount = 0

    print('Populating Arrays')

    for filename in os.listdir(jsonDir):
        with open(os.path.join(jsonDir, filename), 'r') as inputFile:
            currentJson = json.loads(inputFile.read())
            if currentJson['identifier'] not in validList:
                print('Passing over ' + currentJson['identifier'])
                continue
            for mol in currentJson['molecules']:
                if mol['mol_name'] == "New Points":
                    for r in mol['residues']:
                        for a in r['atoms']:
                            if (a['atom_type'] == 'N'):
                                # Standardized
                                Xs[index] = (a['relative_affinity'] - statList[fileCount][0]) / statList[fileCount][1]
                                # Normalized
                                Xn[index] = (a['relative_affinity'] / statList[fileCount][2])
                                # Odd
                                Xo[index] = (a['relative_affinity'] / statList[fileCount][0])
                                # Raw
                                Xr[index] = a['relative_affinity']
                                if a['true_positive']:
                                    y[index] = 1
                                else:
                                    y[index] = 0
                                index += 1
        fileCount += 1

    Xr = Xr.reshape(-1, 1)
    Xn = Xn.reshape(-1, 1)
    Xo = Xo.reshape(-1, 1)
    Xs = Xs.reshape(-1, 1)
    y = label_binarize(y, classes=[0, 1])

    np.savez_compressed(os.path.join('data/', 'roc'), a=Xr, b=y, c=Xn, d=Xs, e=Xo)

def genROC(X, y):
    print('Building model')

    # # # shuffle and split training and test sets
    # # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1,
    # #                                                     random_state=0)

    n_classes = y.shape[1]
    random_state = np.random.RandomState(1)

    print(X.max())
    print(X.min())
    print(y.max())
    print(y.min())

    X_train = X
    X_test = X
    y_train = y
    y_test = y
    # Learn to predict each class against the other
    classifier = linear_model.SGDClassifier(loss='hinge', n_jobs=12, max_iter=1000, random_state=random_state)
    y_score = classifier.fit(X_train, y_train).decision_function(X_test)

    print(classifier.coef_)
    print(classifier.intercept_)

    print('Calculate ROC')

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    print(n_classes)
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:], y_score[:])
        roc_auc[i] = auc(fpr[i], tpr[i])
        p = classifier.predict(X_test)
        tn, fp, fn, tp = confusion_matrix(y_test, p).ravel()
        print(confusion_matrix(y_test, p))
        print('Accuracy: ' + str(accuracy_score(y_test, p)))
        print('Precision: ' + str(precision_score(y_test, p)))
        print('Sensitivity: ' + str(recall_score(y_test, p)))
        print('Specificity: ' + str(tn/(tn+fp)))
        print('MCC: ' + str(matthews_corrcoef(y_test, p)))
        print('AUC: ' + str(auc(fpr[i], tpr[i])))
        print('True Positive: ' + str(tp))
        print('False Positive: ' + str(fp))
        print('True Negative: ' + str(tn))
        print('False Negative: ' + str(fn))


    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    print('Show ROC')

    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange',
            lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

def genHist(X, y, outputDirectory, numBins=500):
    print('Finding Splits')
    pCount = 0
    nCount = 0
    for isPositive in y:
        if isPositive == 1:
            pCount += 1
        else:
            nCount += 1

    print('Splitting Data')

    pScores = np.empty(pCount, dtype=np.float32)
    nScores = np.empty(nCount, dtype=np.float32)

    pIndex = 0
    nIndex = 0

    for i in range(0, X.shape[0]):
        if y[i] == 1:
            pScores[pIndex] = X[i]
            pIndex += 1
        else:
            nScores[nIndex] = X[i]
            nIndex += 1

    print("Making Histogram")

    maxScore = pScores.max()
    check = nScores.max()#to avoid repeat reduction
    if maxScore < check:
        maxScore = check

    print('Max Score: ' + str(maxScore))

    binSpace = np.linspace(0,maxScore,numBins+1)
    # binSpace = np.linspace(0,1,numBins+1)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    pnScores_output = outputDirectory + "/" + timestr + "_posAndNegScores_hist.png"

    figPN = plt.figure()
    plt.xlabel("Score")
    plt.ylabel("Occurance")
    plt.title("Positive and Negative Score Occurance")
    pN, pBins, pPatches = plt.hist(pScores, binSpace, color='b', alpha=0.5, label='Positive')
    nN, nBins, nPatches = plt.hist(nScores, binSpace, color='r', alpha=0.5, label='Negative')
    plt.legend(loc='upper right')
    figPN.savefig(pnScores_output)

    plt.draw()
    plt.pause(1)

    input("<Hit Enter To Close>")
    plt.close(figPN)

def main():
    jsonDir = "data/outputJsons/"
    histDir = "data/hist/"

    prepareData(jsonDir)
    
    archive = np.load(os.path.join('data/', 'roc.npz'))
    X = archive['a']
    y = archive['b']

    genROC(X, y)

    genHist(X, y, histDir)


if __name__ == '__main__':
    main()
