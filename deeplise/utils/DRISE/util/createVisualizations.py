import json
import os

visualizationDir = "data/sites"
jsonDir = "data/predJsons"

def createVisualization(proteinJson, withSurf=False):
    with open(os.path.join(visualizationDir, proteinJson['identifier'] + ".pdb"), 'w+') as visualizationFile:
        if withSurf:
            for mol in proteinJson['molecules']:
                if mol['mol_type'] == "protein":
                    for res in mol['residues']:
                        for atom in res['atoms']:
                            atomString = createAtomString(atom, res)
                            visualizationFile.write(atomString)
        else:
            for mol in proteinJson['molecules']:
                if mol['mol_type'] == "protein":
                    for res in mol['residues']:
                        for atom in res['atoms']:
                            if atom['surf']:
                                atomString = createAtomString(atom, res)
                                visualizationFile.write(atomString)

def createAtomString(atom, res):
    atomString = "ATOM  "

    tempString = str(atom['atom_id'])
    while (len(tempString) < 5):
        tempString = " " + tempString
    atomString += tempString

    tempString = str(atom['atom_name'])
    while (len(tempString) < 4):
        tempString = " " + tempString
    atomString += tempString

    atomString += "A"

    tempString = str(res['res_name'])
    while (len(tempString) < 3):
        tempString = " " + tempString
    atomString += tempString + " "

    tempString = str(res['res_id'])
    while (len(tempString) < 4):
        tempString = " " + tempString
    atomString += tempString

    atomString += "A   "

    tempString = str(round(atom['x'], 3))
    while (len(tempString) < 8):
        tempString = " " + tempString
    atomString += tempString

    tempString = str(round(atom['y'], 3))
    while (len(tempString) < 8):
        tempString = " " + tempString
    atomString += tempString

    tempString = str(round(atom['z'], 3))
    while (len(tempString) < 8):
        tempString = " " + tempString
    atomString += tempString

    tempString = str(round(100.3658, 2))
    while (len(tempString) < 6):
        tempString = " " + tempString
    atomString += tempString

    tempString = str(
        round(atom['relative_affinity'], 2))
    while (len(tempString) < 6):
        tempString = " " + tempString
    atomString += tempString

    atomString += "          "

    tempString = str(atom['element'])
    while (len(tempString) < 2):
        tempString = " " + tempString
    atomString += tempString

    tempString = str(atom['charge'])
    while (len(tempString) < 2):
        tempString = " " + tempString
    atomString += tempString

    atomString += "\n"

    return atomString



if __name__ == '__main__':

    proteinNumber = 0

    for filename in os.listdir(jsonDir):

        if proteinNumber % 25 == 0:
            print(proteinNumber)

        if '.json' not in filename:
            continue

        with open(os.path.join(jsonDir, filename), 'r') as jsonFile:
            proteinJson = json.load(jsonFile)
            createVisualization(proteinJson)
        
        proteinNumber += 1
