from argparse import ArgumentParser
import json
from math import sqrt, floor, ceil
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import numpy as np
import gc

def generateSurfLists(currentJson):
    dnaList = []
    proteinList = []

    for mol in currentJson['molecules']:
        if mol['mol_type'] == 'dna':
            for res in mol['residues']:
                for atom in res['atoms']:
                    tempAtom = atom
                    tempAtom['atom_type'] = res['res_name']
                    dnaList.append(tempAtom)
    
        else:
            for res in mol['residues']:
                for atom in res['atoms']:
                    if atom['surf']:
                        proteinList.append(atom)
                    else:
                        temp = atom
                        temp['atom_type'] = 'None-Surf'
                        proteinList.append(temp)

    return dnaList, proteinList

def generateFullLists(currentJson):
    dnaList = []
    proteinList = []

    for mol in currentJson['molecules']:
        if mol['mol_type'] == 'dna':
            for res in mol['residues']:
                for atom in res['atoms']:
                    tempAtom = atom
                    tempAtom['atom_type'] = res['res_name']
                    dnaList.append(tempAtom)
    
        else:
            for res in mol['residues']:
                for atom in res['atoms']:
                    proteinList.append(atom)

    return dnaList, proteinList

def findTypes(jsonDir):

    atomTypes = []
    hasOddType = False

    for filename in os.listdir(jsonDir):
        with open(os.path.join(jsonDir, filename), 'r') as inputFile:
            currentJson = json.loads(inputFile.read())

            dnaList, proteinList = generateSurfLists(currentJson)
            
            for p in proteinList:
                if p['atom_type'] == '---':
                    hasOddType = True
                    print(p['atom_name'])
                if p['atom_type'] not in atomTypes:
                    atomTypes.append(p['atom_type'])
        if hasOddType:
            print(filename)
            hasOddType = False
                                
    with open('atomTypes.csv', 'w', newline='') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerow(atomTypes)

def toPDBCoords(sphere, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize):
    sphere[1] = (sphere[1] / xOffset) + minX - ceil(bufferSize)
    sphere[2] = (sphere[2] / yOffset) + minY - ceil(bufferSize)
    sphere[3] = (sphere[3] / zOffset) + minZ - ceil(bufferSize)
    return sphere

def toGridCoords(sphere, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize):
    sphere[1] = (sphere[1] - minX + ceil(bufferSize)) * xOffset
    sphere[2] = (sphere[2] - minY + ceil(bufferSize)) * yOffset
    sphere[3] = (sphere[3] - minZ + ceil(bufferSize)) * zOffset
    return sphere

def showGrid(pList, sphereGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, start, stop):

    print('Start: ' + str(start))
    print('Stop: ' + str(stop))
    
    for a in range(start, min(stop, sphereGrid.shape[0])):
        for b in range(sphereGrid.shape[1]):
            for c in range(sphereGrid.shape[2]):
                point = [sphereGrid[a, b, c, 0], a, b, c]
                if point[0] > 0:
                    tp = False
                    if sphereGrid[a, b, c, 1] != 0:
                        tp = True
                    point = toPDBCoords(point, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize)
                    pointDict = {
                                "atom_id" : 1,
                                "atom_name" : "N",
                                "atom_type" : "N",
                                "beta_factor" : point[0],
                                "charge" : 0,
                                "element" : "N",
                                "relative_affinity" : point[0],
                                "surf" : True,
                                "true_positive" : tp,
                                "x" : point[1],
                                "y" : point[2],
                                "z" : point[3]
                                }
                    pList.append(pointDict)
    return pList

def showGridRev(pList, sphereGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, start, stop):

    print('Start: ' + str(start))
    print('Stop: ' + str(stop))
    
    for a in range(sphereGrid.shape[0]-1, -1, -1):
        for b in range(sphereGrid.shape[1]-1, -1, -1):
            for c in range(sphereGrid.shape[2]-1, -1, -1):
                point = [sphereGrid[a, b, c, 0], a, b, c]
                if point[0] > 0:
                    tp = False
                    if sphereGrid[a, b, c, 1] != 0:
                        tp = True
                    point = toPDBCoords(point, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize)
                    pointDict = {
                                "atom_id" : 1,
                                "atom_name" : "N",
                                "atom_type" : "N",
                                "beta_factor" : point[0],
                                "charge" : 0,
                                "element" : "N",
                                "relative_affinity" : point[0],
                                "surf" : True,
                                "true_positive" : tp,
                                "x" : point[1],
                                "y" : point[2],
                                "z" : point[3]
                                }
                    pList.append(pointDict)
    return pList

def thinGrid(pList, sphereGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, distFromProtein):

    for a in range(sphereGrid.shape[0]):
        for b in range(sphereGrid.shape[1]):
            for c in range(sphereGrid.shape[2]):
                nearS = False
                nearNS = False
                point = [sphereGrid[a, b, c, 0], a, b, c]
                point = toPDBCoords(point, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize)
                for p in pList:
                    if p["surf"]:
                        x = point[1] - p['x']
                        y = point[2] - p['y']
                        z = point[3] - p['z']
                        dist = sqrt((x*x)+(y*y)+(z*z))
                        if (dist < distFromProtein) and (dist > distFromProtein-1.0):
                            nearS = True
                    else:
                        x = point[1] - p['x']
                        y = point[2] - p['y']
                        z = point[3] - p['z']
                        dist = sqrt((x*x)+(y*y)+(z*z))
                        if dist < distFromProtein:
                            nearNS = True
                    if nearS:
                        if nearNS:
                            break

                if nearS == False:
                    sphereGrid[a, b, c, 0] = 0.0
                    sphereGrid[a, b, c, 1] = 0.0
                elif nearNS:
                    sphereGrid[a, b, c, 0] = 0.0
                    sphereGrid[a, b, c, 1] = 0.0

    return sphereGrid

def calculateStats(cGrid):
    t = cGrid[:, :, :, 0].flatten()
    m = cGrid[:, :, :, 1].flatten()

    truePos = 0
    falsePos = 0
    trueNeg = 0
    falseNeg = 0

    for i in range(0, t.size):
        if t[i] > 0:
            if t[i] < 0.5:
                if m[i] == 0:
                    trueNeg += 1
                else:
                    falseNeg += 1
            else:
                if m[i] == 1:
                    truePos += 1
                else:
                    falsePos += 1

    stats = {}
    stats['True Positive'] = truePos
    stats['False Positive'] = falsePos
    stats['True Negative'] = trueNeg
    stats['False Negative'] = falseNeg
    
    return stats


def visualizeProc(tasks_to_accomplish, tasks_that_are_done, jsonDir, gridDir, visualizationDir, hparams, bufferSize, intDist):

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
                dnaList, proteinList = generateSurfLists(currentJson)

                minX = proteinList[0]['x']
                maxX = minX
                minY = proteinList[0]['y']
                maxY = minX
                minZ = proteinList[0]['z']
                maxZ = minZ

                for i in range(len(proteinList)):
                    X = proteinList[i]['x']
                    Y = proteinList[i]['y']
                    Z = proteinList[i]['z']

                    minX = min(minX, X)
                    maxX = max(maxX, X)

                    minY = min(minY, Y)
                    maxY = max(maxY, Y)

                    minZ = min(minZ, Z)
                    maxZ = max(maxZ, Z)

                xDim = ceil(maxX - minX + 2*ceil(bufferSize))
                yDim = ceil(maxY - minY + 2*ceil(bufferSize))
                zDim = ceil(maxZ - minZ + 2*ceil(bufferSize))

                xOffset = xDim / (maxX - minX + 2*ceil(bufferSize))
                yOffset = yDim / (maxY - minY + 2*ceil(bufferSize))
                zOffset = zDim / (maxZ - minZ + 2*ceil(bufferSize))

                targetGrid = np.zeros((xDim, yDim, zDim), dtype=np.float32)
                
                try:
                    grids = np.load(gridDir + filename + '.npz')
                except:
                    print(filename + ' not found')
                    continue
                targetGrid = grids['a']
                maskGrid = grids['b']

                currentSize = targetGrid.shape
                combinedGrid = np.empty((currentSize[0], currentSize[1], currentSize[2], 2), dtype=np.float32)
                combinedGrid[:, :, :, 0] = targetGrid
                combinedGrid[:, :, :, 1] = maskGrid
                    
                combinedGrid = thinGrid(proteinList, combinedGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, 4)
                np.savez_compressed(os.path.join('data/', 'combinedGrid.npz'), a=combinedGrid.squeeze())
                combinedGrid = np.load(os.path.join('data/', 'combinedGrid.npz'))['a']

                stats = calculateStats(combinedGrid)
                currentJson['Stats'] = stats

                # totalSize = combinedGrid.shape[0] * combinedGrid.shape[1] * combinedGrid.shape[2]
                # numSlices = ceil(totalSize/250000)
                # sliceSize = ceil(combinedGrid.shape[0]/numSlices)
                
                # for i in range(numSlices):
                #     print('Slice: ' + str(i))
                #     start = sliceSize*i
                #     stop = sliceSize*(i+1)
                #     proteinList = showGrid(proteinList, combinedGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, start, stop)

                proteinList = showGrid(proteinList, combinedGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, 0, 100)
                # proteinList = showGridRev(proteinList, combinedGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, 0, 100)

                maxBeta = 0
                for p in proteinList:
                    if p['beta_factor'] > maxBeta:
                        maxBeta = p['beta_factor']

                if maxBeta == 0:
                    print('WARNING: maxBeta is zero on ' + filename)
                    maxBeta = 1

                newJson = currentJson
                currentAtom = 0

                molId = 0

                for mol in newJson['molecules']:
                    molId = mol['mol_id']
                    if mol['mol_type'] == 'protein':
                        for res in mol['residues']:
                            for atom in res['atoms']:
                                if atom['surf']:
                                    atom['beta_factor'] = 0
                                    atom['relative_affinity'] = atom['beta_factor']
                                    atom['true_positive'] = proteinList[currentAtom]['true_positive']
                                    currentAtom += 1
                                else:
                                    atom['beta_factor'] = 0.0
                                    atom['relative_affinity'] = 0.0
                                    atom['true_positive'] = False

                newPoints = {
                            "identifier": "4r55",
                            "mol_id": molId+1,
                            "mol_name": "New Points",
                            "mol_type": "protein",
                            "residues": [
                                {
                                "atoms": [],
                                "chain": 0,
                                "insertion": 0,
                                "relative_affinity": 0.0,
                                "res_id": 0,
                                "res_name": "SER",
                                "res_type": "protein"
                                }
                            ]}

                for i in range(currentAtom, len(proteinList)):
                    proteinList[currentAtom]['atom_id'] = currentAtom + 1
                    proteinList[currentAtom]['beta_factor'] = (proteinList[currentAtom]['beta_factor'] / 1.0)
                    proteinList[currentAtom]['relative_affinity'] = proteinList[currentAtom]['beta_factor']
                    newPoints['residues'][0]['atoms'].append(proteinList[currentAtom])
                    currentAtom += 1

                newJson['molecules'].append(newPoints)

                print('Outputing Json')
                
                with open(os.path.join(visualizationDir, filename), 'w') as fp:
                    json.dump(newJson, fp, indent=1)
                 
            tasks_that_are_done.put(filename + ' is done by ' + current_process().name)
            # time.sleep(.5)
    return True

def visualize(jsonDir, gridDir, visualizationDir, hparams, bufferSize, intDist, num_workers=1):
    number_of_processes = num_workers
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    # print("Training")
    for filename in os.listdir(jsonDir):
        tasks_to_accomplish.put(filename)

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=visualizeProc, args=(tasks_to_accomplish, tasks_that_are_done, jsonDir, gridDir, visualizationDir, hparams, bufferSize, intDist))
        processes.append(p)
        p.start()

    # completing process
    for p in processes:
        p.join()

    # print the output
    proteinNum = 0
    while not tasks_that_are_done.empty():
        print("Protein Num: " + str(proteinNum))
        print(tasks_that_are_done.get())
        proteinNum += 1

def main(hparams):

    bufferSize = 3
    intDist = 6
    visualizationDir = "data/outputJsons/"
    jsonDir = "data/inputJsons/"
    gridDir = "data/predictions/"

    visualize(jsonDir, gridDir, visualizationDir, hparams, bufferSize, intDist, num_workers=12)

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
