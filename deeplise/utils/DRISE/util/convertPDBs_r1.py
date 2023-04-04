import os
import Bio
import Bio.PDB as pdb
import json

pdbDir = "data/PRNA_full/"
bindingTargets = ['DT', 'DA', 'DC', 'DG', 'DU', ' T', ' A', ' C', ' G', ' U']
mainProteinResidues = ['ALA', 'ARG', 'ASN', 'ASP', 'ASX', 'CYS', 'GLU', 'GLN', 'GLX', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL']
jsonDirectory = "data/inputJsons"

def hasTarget(structure, targetList):
    for model in structure:
        for chain in model:
            for residue in chain:
                r = residue.resname
                if r[0] == ' ':
                    for target in targetList:
                        if target == r[1:]:
                            return True
    return False

def removeNoTargetPDBs():

    parser = pdb.PDBParser(PERMISSIVE=False, QUIET=True)

    count = 0

    for filename in os.listdir(pdbDir):
        if '.pdb' not in filename:
            continue
        # print('Found PDB: ', filename)

        count += 1

        if count%100 == 0:
            print(count)

        noDNA = False

        with open(os.path.join(pdbDir, filename), 'r') as pdbFile:
            try:
                pdbStructure = parser.get_structure(filename[:-4], pdbFile)
                if hasTarget(pdbStructure, bindingTargets):
                    continue
                else:
                    print('No DNA')
                    noDNA = True
            except:
                print(filename + " contains errors")

        if noDNA:
            os.remove(os.path.join(pdbDir, filename))

def getParentMolecule(mainDict, chainID):
    for key in mainDict['header']['compound']:
        for cID in mainDict['header']['compound'][key]['chain']:
            if chainID.lower() == cID.lower():
                return int(key) - 1
    
    while(True):
        print("Error")



def convertPDBtoJSON():
    parser = pdb.PDBParser(PERMISSIVE=False, get_header=True, QUIET=True)

    count = 0

    atomTypes = {}

    for filename in os.listdir(pdbDir):
        if count == 9999999:
            break
        elif count%50 == 0:
            print(count)
        if '.pdb' not in filename:
            continue

        pdbName = filename[:-4]
        data = {}

        data['identifier'] = pdbName

        try:

            with open(os.path.join(pdbDir, filename), 'r') as pdbFile:

                pdbStructure = parser.get_structure(pdbName, pdbFile)

                data['molecules'] = []
                data['header'] = pdbStructure.header

                numModels = 0

                for model in pdbStructure:

                    numModels += 1
                    if numModels > 1:
                        break

                    moleculeList = []
                    moleculeID = 0

                    for key in data['header']['compound']:

                        compoundDict = {}

                        compoundDict['identifier'] = pdbName
                        compoundDict['mol_id'] = moleculeID
                        compoundDict['mol_name'] = data['header']['compound'][key]['molecule']
                        compoundDict['mol_type'] = 'protein'
                        compoundDict['residues'] = []

                        moleculeList.append(compoundDict)

                        moleculeID += 1

                    data['molecules'] = moleculeList

                    atomID = 0

                    for chain in model:
                        
                        # print(chain.id)
                        parentMolecule = getParentMolecule(data, chain.id)

                        residueList = []

                        for residue in chain:

                            residueDict = {}
                            residueDict['atoms'] = []
                            residueDict['chain'] = chain.id
                            # residueDict['insertion'] = residue.insertion
                            residueDict['relative_affinity'] = 0
                            residueDict['res_id'] = residue.id[1]
                            residueDict['res_name'] = residue.resname.strip()
                            residueDict['res_type'] = 'protein'

                            r = residue.resname
                            if r[0] == ' ':
                                for target in bindingTargets:
                                    if target == r[1:]:
                                        if r[1] == ' ':
                                            residueDict['res_type'] = 'rna'
                                            data['molecules'][parentMolecule]['mol_type'] = 'rna'
                                        else:                                            
                                            residueDict['res_type'] = 'dna'
                                            data['molecules'][parentMolecule]['mol_type'] = 'dna'
                                        break

                            atomList = []

                            for atom in residue:

                                atomID += 1

                                atomDict = {}
                                atomDict['atom_id'] = atomID
                                atomDict['atom_name'] = atom.id
                                atomDict['atom_type'] = '---'
                                atomDict['beta_factor'] = atom.bfactor
                                atomDict['charge'] = 0
                                atomDict['element'] = atom.element
                                atomDict['relative_affinity'] = 0
                                atomDict['surf'] = False
                                atomDict['true_positive'] = False

                                coordinates = atom.coord

                                atomDict['x'] = float(coordinates[0])
                                atomDict['y'] = float(coordinates[1])
                                atomDict['z'] = float(coordinates[2])

                                atomList.append(atomDict)

                                if residueDict['res_name'] in mainProteinResidues:

                                    currentType = atom.id + ',' + residueDict['res_name'] + ','
                                    currentType = currentType + atom.id + "-" + residueDict['res_name'] + ','

                                    if residueDict['res_name'] in atomTypes.keys():

                                        if currentType not in atomTypes[residueDict['res_name']]:

                                            atomTypes[residueDict['res_name']].append(currentType)
                                    else:

                                        atomTypes[residueDict['res_name']] = [currentType]

                            residueDict['atoms'] = atomList

                            residueList.append(residueDict)
                        if 'residues' in data['molecules'][parentMolecule]:
                            for r in residueList:
                                data['molecules'][parentMolecule]['residues'].append(r)
                        else:
                            data['molecules'][parentMolecule]['residues'] = residueList

                if numModels > 1:
                    print('ERROR: Multiple models detected within file. Only creating the first model JSON for ' + pdbName)
                with open(os.path.join(jsonDirectory, pdbName + ".json"), "w") as write_file:
                    json.dump(data, write_file, indent=1)

                # except:
                #     print(filename + " contains errors") 3021

            count += 1
        except:
            print(filename)

    print("Saving atomTypes to file")

    typeCount = 0

    with open('uniqueAtomTypes.csv', 'w') as f:
        for key in atomTypes.keys():
            currentResidue = atomTypes[key]
            for item in currentResidue:
                typeCount += 1
                f.write("%s\n" % item)

    print("Number of unique atom types found: " + str(typeCount))


def test():
    parser = pdb.PDBParser(PERMISSIVE=True, get_header=True, QUIET=False)

    count = 0

    for filename in os.listdir(pdbDir):
            if count == 1:
                break
            if '.pdb' not in filename:
                continue

            pdbName = filename[:-4]
            data = {}

            data['identifier'] = pdbName

            with open(os.path.join(pdbDir, filename), 'r') as pdbFile:
                try:
                    pdbStructure = parser.get_structure(pdbName, pdbFile)

                    for model in pdbStructure:
                        for chain in model:
                            print(chain.id)
                            # for residue in chain:
                            #     # print(residue.segid)


                except:
                    print(filename + " contains errors")


if __name__ == '__main__':
    convertPDBtoJSON()
    #test()
