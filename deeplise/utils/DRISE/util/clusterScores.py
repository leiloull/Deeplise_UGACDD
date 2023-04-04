import json
import os
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from numpy import mean, std
from math import sqrt

jsonDir = "data/complexJsons"

def determineBindingSite(proteinJson):
    totalAtoms = 0

    for mol in proteinJson['molecules']:
        if mol['mol_type'] == "protein":
            for res in mol['residues']:
                for atom in res['atoms']:
                    if atom['surf']:
                        totalAtoms += 1
    
    proteinArray = np.empty((totalAtoms, 4), dtype=np.float64)

    index = 0

    for mol in proteinJson['molecules']:
        if mol['mol_type'] == "protein":
            for res in mol['residues']:
                for atom in res['atoms']:
                    if atom['surf']:
                        proteinArray[index, 0] = atom["x"]
                        proteinArray[index, 1] = atom["y"]
                        proteinArray[index, 2] = atom["z"]
                        proteinArray[index, 3] = atom["relative_affinity"]
                        index += 1

    proteinArray = (proteinArray - mean(proteinArray, axis=0)) / std(proteinArray, axis=0)
    proteinArray[:, 3] *= 2

    # numClusters = int(totalAtoms/25)

    # if numClusters < 5:
    #     numClusters = 5

    numClusters = 3

    kMeans1 = KMeans(n_clusters=numClusters, n_init=12, n_jobs=12, max_iter=1000, random_state=0).fit(proteinArray)

    groupAverages = np.zeros(numClusters, dtype=np.float64)
    groupCount = np.zeros(numClusters, dtype=np.int32)
    labels1 = kMeans1.labels_

    for atomIndex in range(0, len(labels1)):
        groupAverages[labels1[atomIndex]] += proteinArray[atomIndex, 3]
        groupCount[labels1[atomIndex]] += 1

    groupAverages = groupAverages / groupCount

    kMeans2 = KMeans(n_clusters=2, n_init=12, n_jobs=12, max_iter=1000, random_state=0).fit(groupAverages.reshape(-1, 1))
    
    labels2 = kMeans2.labels_

    for atomIndex in range(0, len(labels1)):
        labels1[atomIndex] = labels2[labels1[atomIndex]]

    boundGroup = 0

    if kMeans2.cluster_centers_[1] > kMeans2.cluster_centers_[0]:
        boundGroup = 1

    atomIndex = 0

    for mol in proteinJson['molecules']:
        if mol['mol_type'] == "protein":
            for res in mol['residues']:
                for atom in res['atoms']:
                    if atom['surf']:
                        if labels1[atomIndex] == boundGroup:
                            atom['relative_affinity'] = 1.0
                        else:
                            atom['relative_affinity'] = 0.0
                        atomIndex += 1

    return proteinJson


def bindingStatistics(proteinJson):
    totalAtoms = 0
    totalBoundAtoms = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0

    for mol in proteinJson['molecules']:
        if mol['mol_type'] == "protein":
            for res in mol['residues']:
                for atom in res['atoms']:
                    if atom['surf']:
                        if atom['true_positive']:
                            if atom['relative_affinity'] == 1:
                                truePositive += 1
                            else:
                                falseNegative += 1
                            totalBoundAtoms += 1
                        else:
                            if atom['relative_affinity'] == 0:
                                trueNegative += 1
                            else:
                                falsePositive += 1
                        totalAtoms += 1

    statArray = np.empty(6, dtype=np.int64)
    statArray[0] = totalAtoms
    statArray[1] = totalBoundAtoms
    statArray[2] = truePositive
    statArray[3] = falsePositive
    statArray[4] = trueNegative
    statArray[5] = falseNegative

    return statArray         


if __name__ == '__main__':

    proteinNumber = 0

    for filename in os.listdir(jsonDir):
        if '.json' not in filename:
            continue
        proteinNumber += 1

    globalStatArray = np.zeros((proteinNumber, 6), dtype=np.int64)
    proteinNumber = 0

    for filename in os.listdir(jsonDir):

        if proteinNumber % 25 == 0:
            print(proteinNumber)

        if '.json' not in filename:
            continue

        with open(os.path.join(jsonDir, filename), 'r') as jsonFile:
            proteinJson = json.load(jsonFile)
            proteinJson = determineBindingSite(proteinJson)
            proteinStats = bindingStatistics(proteinJson)
            globalStatArray[proteinNumber, :] = proteinStats
            with open(os.path.join('data/predJsons', filename + ".json"), "w") as write_file:
                json.dump(proteinJson, write_file, indent=1)

        proteinNumber += 1

    globalStats = np.sum(globalStatArray, axis=0)

    print("Global Statistics")
    totalAtoms = globalStats[0]
    totalBindingAtoms = globalStats[1]
    tP = globalStats[2]
    fP = globalStats[3]
    tN = globalStats[4]
    fN = globalStats[5]

    print("Binding Rate: " + str(totalBindingAtoms/totalAtoms))
    print("Accuracy: " + str((tP + tN)/totalAtoms))
    print("Sensitivity: " + str(tP/(tP + fN)))
    print("Specificity: " + str(tN/(fP + tN)))

    mN = tN + tP + fN + fP
    mS = (tP + fN)/mN
    mP = (tP + fP)/mN

    print("MCC: " + str(((tP/mN) - (mS*mP))/sqrt(mP*mS*(1-mS)*(1-mP))))

    individualStats = np.zeros((proteinNumber, 5), dtype=np.float64)
    
    currentAtom = 0
    numNoBind = 0

    for row in globalStatArray:
        totalAtoms =row[0]
        totalBindingAtoms = row[1]
        tP = row[2]
        fP = row[3]
        tN = row[4]
        fN = row[5]

        individualStats[currentAtom, 0] += totalBindingAtoms/totalAtoms
        individualStats[currentAtom, 1] += (tP + tN)/totalAtoms
        if (tP+fN) == 0:
            # print("No Positives Detected")
            numNoBind += 1
            individualStats[currentAtom, 2] = 1
            individualStats[currentAtom, 4]=((tP*tN)-(fP*fN))/sqrt((tP+fP)*(1)*(tN+fP)*(tN+fN))
        else:
            individualStats[currentAtom, 2] += tP/(tP + fN)
            individualStats[currentAtom, 4]=((tP*tN)-(fP*fN))/sqrt((tP+fP)*(tP+fN)*(tN+fP)*(tN+fN))
        individualStats[currentAtom, 3] += tN/(fP + tN)
        currentAtom += 1

    averageMean = np.mean(individualStats, axis=0)
    averageStd = np.std(individualStats, axis=0)

    print("\nAverage Statistics per Protein")
    print("Binding Rate: " + str(averageMean[0]) + " with std: " + str(averageStd[0]))
    print("Accuracy: " + str(averageMean[1]) + " with std: " + str(averageStd[1]))
    print("Sensitivity: " + str(averageMean[2]) + " with std: " + str(averageStd[2]))
    print("Specificity: " + str(averageMean[3]) + " with std: " + str(averageStd[3]))
    print("MCC: " + str(averageMean[4]) + " with std: " + str(averageStd[4]))
    print("Number of proteins with no bound atoms: " + str(numNoBind))

    

            
