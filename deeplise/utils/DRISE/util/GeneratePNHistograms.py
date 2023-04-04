#!/usr/bin/env python3

#Author: Jackson Parker

import sys, getopt
import os
import json
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import time

def histOverlap(h1,h2,bins):
    overlap = []
    sumOverlap = 0
    for i in range(bins):
        print(i,-h1[i],h2[i])
        if h1[i] > h2[i]:
            overlap.append(-1-h2[i])
            sumOverlap += abs(h2[i])
        elif h1[i] < h2[i]:
            overlap.append(1+h1[i])
            sumOverlap += abs(h1[i])
        else:
            overlap.append(0)
    return sumOverlap, overlap

def getPNR_residue(jsonDirectory, normalize):
    filesUsed = []
    ratios = []
    pScores = []
    nScores = []
    for filename in os.listdir(jsonDirectory):
        if '.json' not in filename:
            continue
        print('Found JSON: ', filename)
        with open(jsonDirectory + filename) as json_data:
            d = json.load(json_data)
            try:
                if len(d['molecules']) == 0: continue
                positives = 0
                current_pScores = []
                negatives = 0
                current_nScores = []
                for mol in d['molecules']:
                    if mol['mol_type'] == 'protein':
                        for res in mol['residues']:
                            isTruth = False
                            resScore = res['relative_affinity']
                            for atom in res['atoms']:
                                if atom['true_positive']:
                                    isTruth = True
                                    break
                            if isTruth:
                                positives += 1
                                current_pScores.append(resScore)
                            else:
                                negatives += 1
                                current_nScores.append(resScore)
                ratio = 0.0
                if negatives != 0:
                    ratio = float(positives)/float(negatives)
                    ratios.append(ratio)
                    current_pScores = np.array(current_pScores, dtype=np.float32)
                    for val in current_pScores:
                        if normalize:
                            val -= current_pScores.min()
                            val /= current_pScores.ptp()
                        pScores.append(val)
                    current_nScores = np.array(current_nScores, dtype=np.float32)
                    for val in current_nScores:
                        if normalize:
                            val -= current_nScores.min()
                            val /= current_nScores.ptp()
                        nScores.append(val)
                    filesUsed.append(filename)
            except KeyError: continue
    return ratios, pScores, nScores, filesUsed

def getPNR_atomic(jsonDirectory, normalize):
    filesUsed = []
    ratios = []
    pScores = []
    nScores = []
    for filename in os.listdir(jsonDirectory):
        if '.json' not in filename:
            continue
        print('Found JSON: ', filename)
        with open(jsonDirectory + filename) as json_data:
            d = json.load(json_data)
            try:
                if len(d['molecules']) == 0: continue
                positives = 0
                current_pScores = []
                negatives = 0
                current_nScores = []
                for mol in d['molecules']:
                    if mol['mol_type'] == 'protein':
                        for res in mol['residues']:
                            if len(res['atoms']) == 0: continue
                            for atom in res['atoms']:
                                if atom['true_positive']:
                                    positives += 1
                                    current_pScores.append(atom['relative_affinity'])
                                else:
                                    negatives += 1
                                    current_nScores.append(atom['relative_affinity'])
                ratio = 0.0
                if negatives != 0:
                    ratio = float(positives)/float(negatives)
                    ratios.append(ratio)
                    current_pScores = np.array(current_pScores, dtype=np.float32)
                    for val in current_pScores:
                        if normalize:
                            val -= current_pScores.min()
                            val /= current_pScores.ptp()
                        pScores.append(val)
                    current_nScores = np.array(current_nScores, dtype=np.float32)
                    for val in current_nScores:
                        if normalize:
                            val -= current_nScores.min()
                            val /= current_nScores.ptp()
                        nScores.append(val)
                    filesUsed.append(filename)
            except KeyError: continue
    return ratios, pScores, nScores, filesUsed

def parseArgs(argv):
    normalize = False
    input = ''
    output = ''
    type = "residue"
    numBins = 20
    try:
        opts, args = getopt.getopt(argv, "nhi:o:f:b:",["ifile=","ofile=","focus=","bins="])
    except getopt.GetoptError:
        print('GeneratePNHistograms.py -i <input directory> -o <outputfile>')
        sys.exit(2)
    if len(opts) == 0:
        print('GeneratePNHistograms.py -i <input directory> -o <outputfile>')
        sys.exit(0)
    for opt, arg in opts:
        if opt == '-h':
            print('GeneratePNHistograms.py -i <input directory> -o <outputfile>')
            sys.exit(0)
        elif opt in ("-i", "--ifile"):
            input = arg
        elif opt in ("-o","--ofile"):
            output = arg
        elif opt in ("-f","--focus"):
            if arg == "residue" or arg == "atomic":
                type = arg
            else:
                print("Error focus type can only be residue or atomic")
                sys.exit(-1)
        elif opt in ("-b","--bins"):
            numBins = int(arg)
        elif opt == '-n':
            normalize = True

    print('Directory of ISE jsons is ', input)
    if output == '':
        output = "./"
    print('Output Directory is ', output)
    print('Level of focus is ', type)
    return input, output, type, normalize, numBins

def main(argv):
    jsonDirectory = ''
    outputDirectory = ''
    filesUsed = []
    ratios = []
    pScores = []
    nScores = []
    type = "residue"
    normalize = False
    numBins = 20

    jsonDirectory, outputDirectory, type, normalize, numBins = parseArgs(argv)

    if type == "residue":
        ratios, pScores, nScores, filesUsed = getPNR_residue(jsonDirectory, normalize)
    else:
        ratios, pScores, nScores, filesUsed = getPNR_atomic(jsonDirectory, normalize)

    pScores = np.array(pScores)
    nScores = np.array(nScores)
    ratios = np.array(ratios)

    maxScore = pScores.max()
    check = nScores.max()#to avoid repeat reduction
    if maxScore < check:
        maxScore = check

    binSpace = np.linspace(0,maxScore,numBins+1)

    timestr = time.strftime("%Y%m%d-%H%M%S")

    ratio_output = outputDirectory + "/" + timestr + "_PNRatio_hist.png"
    pnScores_output = outputDirectory + "/" + timestr + "_posAndNegScores_hist.png"
    overlap_output = outputDirectory + "/" + timestr + "_TPTNoverlap_hist.png"

    figPN = plt.figure()
    plt.xlabel("Score")
    plt.ylabel("Occurance")
    plt.title("Positive and Negative Score Occurance - "+ type + "\n# of complexes = "+ str(len(filesUsed)))
    pN, pBins, pPatches = plt.hist(pScores, binSpace, color='b', alpha=0.5, label='Positive')
    nN, nBins, nPatches = plt.hist(nScores, binSpace, color='r', alpha=0.5, label='Negative')
    plt.legend(loc='upper right')
    figPN.savefig(pnScores_output)

    figPNR = plt.figure()
    plt.xlabel("Positive/Negative Ratio")
    plt.ylabel("Occurance")
    plt.title("Positive/Negative Ratio Occurance - " + type + "\n# of complexes = "+ str(len(filesUsed)))
    rN, rBins, rPatches = plt.hist(ratios, bins=numBins, color='b', alpha=0.5)
    figPNR.savefig(ratio_output)

    figPNOverlap = plt.figure()
    sumOverlap, overlap = histOverlap(nN,pN,numBins)
    plt.xlabel("TP TN score")
    plt.ylabel("Similarity")
    plt.title("TP TN score - "+ type + "\ntotal overlap = " + str(sumOverlap) + "|# of complexes = "+ str(len(filesUsed)))
    plt.bar(nBins[1:], height=overlap, edgecolor='black', color='red',width=nBins[1]-nBins[0])

    figPNOverlap.savefig(overlap_output)

    plt.draw()
    plt.pause(1)

    input("<Hit Enter To Close>")
    plt.close(figPN)
    plt.close(figPNR)
    plt.close(figPNOverlap)

if __name__ == "__main__":
    main(sys.argv[1:])
