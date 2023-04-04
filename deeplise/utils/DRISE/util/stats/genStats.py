import os
import json

jsonDir = "data/complexJsons"
triangleJsonDir = "data"

def genAtomStats():

    atomStats = {}
    count = 0

    for filename in os.listdir(jsonDir):
        
        if '.json' not in filename:
            continue
        
        with open(os.path.join(jsonDir, filename), 'r') as jsonFile:

            proteinJson = json.load(jsonFile)

            for mol in proteinJson['molecules']:
                if mol['mol_type'] == "protein":
                    for res in mol['residues']:
                        for atom in res['atoms']:

                            atomType = atom['atom_type']
                        
                            if atomType in atomStats.keys():
                                
                                if atom['true_positive'] == True:
                                    atomStats[atomType]['Positive'] += 1
                                else:
                                    atomStats[atomType]['Negative'] += 1
                            
                            else:

                                atomDict = {}
                                atomDict['Positive'] = 0
                                atomDict['Negative'] = 0
                                atomStats[atomType] = atomDict
        
        count += 1

        if count % 10 == 0:
            print(count)

    totalBind = 0
    totalNonBind = 0

    for key in atomStats.keys():
        totalBind += atomStats[key]['Positive']
        totalNonBind += atomStats[key]['Negative']

    for key in atomStats.keys():

        currentType = atomStats[key]
        currentType['Probability'] = currentType['Positive'] / (currentType['Positive'] + currentType['Negative'])
        currentType['Odds'] = currentType['Positive'] / currentType['Negative']
        currentType['Enrichment Factor'] = (currentType['Positive'] / totalBind) / (currentType['Negative'] / totalNonBind)

        atomStats[key] = currentType
    
    with open("atomStats.json", "w") as write_file:
        json.dump(atomStats, write_file, indent=1)


def genTriangleStats():

    triangleDict = {}

    with open(os.path.join(triangleJsonDir, "triangles.json"), 'r') as jsonFile:

        triangleJson = json.load(jsonFile)

        totalBind = 0
        totalNonBind = 0

        for tri in triangleJson['triangles']:

            totalBind += tri['interactions']
            totalNonBind += tri['occurances'] - tri['interactions']

        for tri in triangleJson['triangles']:

            currentStats = {}

            currentStats['Interactions'] = tri['interactions']
            currentStats['Occurances'] = tri['occurances']

            if tri['occurances'] == 0:
                currentStats['Probability'] = 'N/A'
                currentStats['Odds'] = 'N/A'
                currentStats['Enrichment Factor'] = 'N/A'
            else:
                currentStats['Probability'] = tri['interactions'] / tri['occurances']

                if (tri['occurances'] - tri['interactions']) == 0:
                    currentStats['Odds'] = 'infinite'
                    currentStats['Enrichment Factor'] = 'infinite'
                else:
                    try:
                        currentStats['Odds'] = tri['interactions'] / (tri['occurances'] - tri['interactions'])
                        currentStats['Enrichment Factor'] = (tri['interactions'] / totalBind) / ((tri['occurances'] - tri['interactions']) / totalNonBind)
                    except:
                        print(tri['interactions'])
                        print(totalBind)
                        print((tri['occurances'] - tri['interactions']))
                        print(totalNonBind)


            triangleDict[str(tri['atom_type_1']) + '-' + str(tri['atom_type_2']) + '-' + str(tri['atom_type_3'])] = currentStats

        with open("triangleStats.json", "w") as write_file:
            json.dump(triangleDict, write_file, indent=1)

        with open('triangleStats.csv', 'w') as f:
            header = "ID, Interactions, Occurances, Probability, Odds, Enrichment Factor"
            f.write("%s\n" % header)
            for key in triangleDict.keys():
                currentTriangle = triangleDict[key]
                data = key
                for item in currentTriangle.keys():
                    data += ',' + str(currentTriangle[item])
                f.write("%s\n" % data)

if __name__ == '__main__':
    genTriangleStats()
    genAtomStats()
