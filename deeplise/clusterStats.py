import pandas as pd
import numpy as np
import json
import os
from math import sqrt

def bindingStatistics(proteinJson):
    totalAtoms = 0
    totalBoundAtoms = 0
    truePositive = 0
    falsePositive = 0
    trueNegative = 0
    falseNegative = 0
    numNuc = 0
    numProRes = 0

    for mol in proteinJson['molecules']:
        if mol['mol_type'] == "protein":
            if mol['mol_name'] == "New Points":
                for res in mol['residues']:
                    for atom in res['atoms']:
                        if atom['atom_name'] == 'N' and atom['atom_type'] == 'N':
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
            else:
                for res in mol['residues']:
                    numProRes += 1
        elif mol['mol_type'] == "dna":
            for res in mol['residues']:
                numNuc += 1


    statArray = np.empty(8, dtype=np.int64)
    statArray[0] = totalAtoms
    statArray[1] = totalBoundAtoms
    statArray[2] = truePositive
    statArray[3] = falsePositive
    statArray[4] = trueNegative
    statArray[5] = falseNegative
    statArray[6] = numNuc
    statArray[7] = numProRes

    return statArray

def makeClusterDF(jsonDir, valDir):
    proteinNumber = 0
    nameList = []
    bindRateList = []
    accList = []
    precList = []
    sensitivityList = []
    specificityList = []
    mccList = []
    tpList = []
    fpList = []
    tnList = []
    fnList = []
    nucNumList = []
    nucProResList = []
    valList = []
    
    for filename in os.listdir(jsonDir):

        if proteinNumber % 10 == 0:
            print(proteinNumber)

        if '.json' not in filename:
            continue

        with open(os.path.join(jsonDir, filename), 'r') as jsonFile:
            proteinJson = json.load(jsonFile)
            nameList.append(proteinJson['identifier'])

            proteinStats = bindingStatistics(proteinJson)

            totalAtoms = proteinStats[0]
            totalBindingAtoms = proteinStats[1]
            tP = proteinStats[2]
            fP = proteinStats[3]
            tN = proteinStats[4]
            fN = proteinStats[5]

            tpList.append(tP)
            fpList.append(fP)
            tnList.append(tN)
            fnList.append(fN)

            bindRateList.append(totalBindingAtoms/totalAtoms)
            accList.append((tP + tN)/totalAtoms)
            precList.append(tP/(tP+fP))
            sensitivityList.append(tP/(tP + fN))
            specificityList.append(tN/(fP + tN))

            mN = tN + tP + fN + fP
            mS = (tP + fN)/mN
            mP = (tP + fP)/mN

            mccList.append(((tP/mN) - (mS*mP))/sqrt(mP*mS*(1-mS)*(1-mP)))
            nucNumList.append(proteinStats[6])
            nucProResList.append(proteinStats[7])
            
            isVal = False
            gridName = filename[:7] + '.json-0-0.npz'
            if os.path.isfile(os.path.join(valDir, gridName)):
                print ("Is validation")
                isVal = True
            valList.append(isVal)

        
        proteinNumber += 1
    
    data = {'Name': nameList, 
            'Binding Rate': bindRateList, 
            'Accuracy': accList,
            'Precision': precList,
            'Sensitivity': sensitivityList,
            'Specificity': specificityList,
            'MCC': mccList,
            'True Positives': tpList,
            'False Positives': fpList,
            'True Negatives': tnList,
            'False Negatives': fnList,
            'numNuc': nucNumList,
            'numProRes': nucProResList,
            'isVal': valList}
    
    df = pd.DataFrame.from_dict(data)

    return df

if __name__ == '__main__':

    jsonDir = 'data/predJsons'
    valDir = 'data/val-grids'

    clusterDF = makeClusterDF(jsonDir, valDir)

    csv_data = clusterDF.to_csv('GfG.csv', index = True) 
    print('\nCSV String:\n', csv_data) 

    
