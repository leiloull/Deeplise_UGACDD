import json
import os
from math import sqrt, floor, ceil
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import numpy as np
import csv

def generateSurfLists(currentJson):
    dnaList = []
    proteinList = []

    # numMols = 0
    # numProteinAtom = 0

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
    #                 numProteinAtom += 1
    #     numMols += 1
    #     print(numProteinAtom)
    #     print(len(proteinList))
    # print(numProteinAtom)
    # print(len(proteinList))
    # print(numMols)

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

def showGrid(pList, sphereGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize):
    
    for a in range(sphereGrid.shape[0]):
        for b in range(sphereGrid.shape[1]):
            for c in range(sphereGrid.shape[2]):
                point = [sphereGrid[a, b, c], a, b, c]
                if point[0] > 0:
                    # print(point[0])
                    
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
                                "true_positive" : False,
                                "x" : point[1],
                                "y" : point[2],
                                "z" : point[3]
                                }
                    pList.append(pointDict)
    
    return pList

def trainProc(tasks_to_accomplish, tasks_that_are_done, jsonDir, atomTypes, bufferSize, intDist):
    largest = 0
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

                product = xDim * yDim * zDim
                if product > largest:
                    largest = product

                inputGrid = np.zeros((len(atomTypes), xDim, yDim, zDim), dtype=np.float32)
                targetGrid = np.zeros((1, xDim, yDim, zDim), dtype=np.float32)
                # print(grid.shape)

                for i in range(len(proteinList)):
                    X = (proteinList[i]['x'] - minX + ceil(bufferSize)) * xOffset
                    Y = (proteinList[i]['y'] - minY + ceil(bufferSize)) * yOffset
                    Z = (proteinList[i]['z'] - minZ + ceil(bufferSize)) * zOffset

                    minIndex = [999.0, 0, 0, 0]

                    for a in range(floor(X - bufferSize), ceil(X + bufferSize)):
                        for b in range(floor(Y - bufferSize), ceil(Y + bufferSize)):
                            for c in range(floor(Z - bufferSize), ceil(Z + bufferSize)):
                                dist = sqrt((X - a)**2 + (Y - b)**2 + (Z - c)**2)
                                if dist < minIndex[0]:
                                    minIndex[0] = dist
                                    minIndex[1] = a
                                    minIndex[2] = b
                                    minIndex[3] = c
                    
                    inputGrid[atomTypes.index(proteinList[i]['atom_type']), minIndex[1], minIndex[2], minIndex[3]] = 1

                for i in range(inputGrid.shape[1]):
                    for j in range(inputGrid.shape[2]):
                        for k in range(inputGrid.shape[3]):
                            point = toPDBCoords([0, i, j, k], xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize)

                            for p in proteinList:
                                x = point[1] - p['x']
                                y = point[2] - p['y']
                                z = point[3] - p['z']
                                dist = sqrt(x**2 + y**2 + z**2)
                                if dist < intDist:
                                    if p['atom_type'] != 'None-Surf':
                                        for d in dnaList:
                                            x = point[1] - d['x']
                                            y = point[2] - d['y']
                                            z = point[3] - d['z']
                                            dist = sqrt(x**2 + y**2 + z**2)
                                            distToDNA = dist
                                            if dist < intDist:
                                                x = p['x'] - d['x']
                                                y = p['y'] - d['y']
                                                z = p['z'] - d['z']
                                                dist = sqrt(x**2 + y**2 + z**2)
                                                if dist < intDist:
                                                    if distToDNA <= dist:
                                                        targetGrid[0, i, j, k] = 1
                                                        break

                X = inputGrid
                y = targetGrid

                for i in range(4):
                    tempX = np.rot90(X, i, (1, 2))
                    tempY = np.rot90(y, i, (1, 2))
                    np.savez_compressed(os.path.join('grids', filename + '-0-' + str(i)), a=tempX, b=tempY)

                rotX = np.rot90(X, 1, (2, 3))
                rotY = np.rot90(y, 1, (2, 3))

                for i in range(4):
                    tempX = np.rot90(rotX, i, (1, 2))
                    tempY = np.rot90(rotY, i, (1, 2))
                    np.savez_compressed(os.path.join('grids', filename + '-1-' + str(i)), a=tempX, b=tempY)

                rotX = np.rot90(X, 2, (2, 3))
                rotY = np.rot90(y, 2, (2, 3))

                for i in range(4):
                    tempX = np.rot90(rotX, i, (1, 2))
                    tempY = np.rot90(rotY, i, (1, 2))
                    np.savez_compressed(os.path.join('grids', filename + '-2-' + str(i)), a=tempX, b=tempY)

                rotX = np.rot90(X, 3, (2, 3))
                rotY = np.rot90(y, 3, (2, 3))

                for i in range(4):
                    tempX = np.rot90(rotX, i, (1, 2))
                    tempY = np.rot90(rotY, i, (1, 2))
                    np.savez_compressed(os.path.join('grids', filename + '-3-' + str(i)), a=tempX, b=tempY)

                rotX = np.rot90(X, 1, (1, 3))
                rotY = np.rot90(y, 1, (1, 3))

                for i in range(4):
                    tempX = np.rot90(rotX, i, (1, 2))
                    tempY = np.rot90(rotY, i, (1, 2))
                    np.savez_compressed(os.path.join('grids', filename + '-4-' + str(i)), a=tempX, b=tempY)

                rotX = np.rot90(X, 3, (1, 3))
                rotY = np.rot90(y, 3, (1, 3))

                for i in range(4):
                    tempX = np.rot90(rotX, i, (1, 2))
                    tempY = np.rot90(rotY, i, (1, 2))
                    np.savez_compressed(os.path.join('grids', filename + '-5-' + str(i)), a=tempX, b=tempY)
                
            tasks_that_are_done.put(filename + ' is done by ' + current_process().name)
            # time.sleep(.5)
    return True

def train(jsonDir, atomTypes, bufferSize, intDist, num_workers=1):
    number_of_processes = num_workers
    tasks_to_accomplish = Queue()
    tasks_that_are_done = Queue()
    processes = []

    # print("Training")
    for filename in os.listdir(jsonDir):
        tasks_to_accomplish.put(filename)

    # creating processes
    for w in range(number_of_processes):
        p = Process(target=trainProc, args=(tasks_to_accomplish, tasks_that_are_done, jsonDir, atomTypes, bufferSize, intDist))
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


if __name__ == '__main__':

    jsonDir = "inputJsons"
    bufferSize = 3
    intDist = 6

    if not os.path.isfile('atomTypes.csv'):
        print("Finding atom types")
        findTypes(jsonDir)

    atomTypes = []
    
    with open('atomTypes.csv', newline='') as f:
        reader = csv.reader(f)
        atomTypes = list(reader)[0]

    print(atomTypes)

    train(jsonDir, atomTypes, bufferSize, intDist, num_workers=27)
