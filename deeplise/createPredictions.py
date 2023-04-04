from pytorch_lightning import Trainer
from model import CoolSystem
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser
import torch.nn
import torch.nn.functional as F
from PIL import Image
import torch
from utils.dataset import BasicDataset
from torchvision import transforms
from torchviz import make_dot, make_dot_from_trace
import json
from math import sqrt, floor, ceil
from multiprocessing import Lock, Process, Queue, current_process
import time
import queue # imported for using queue.Empty exception
import csv
import os
import numpy as np

def createVisualization(proteinJson, visualizationDir, withSurf=True):
    with open(os.path.join(visualizationDir, proteinJson['identifier'] + ".pdb"), 'w+') as visualizationFile:
        scoreList = []
        for mol in proteinJson['molecules']:
            if mol['mol_type'] == "protein":
                for res in mol['residues']:
                    for atom in res['atoms']:
                        scoreList.append(atom['relative_affinity'])
        
        npArray = np.array(scoreList)
        mean = npArray.mean()
        std = npArray.std()

        if withSurf:
            for mol in proteinJson['molecules']:
                if mol['mol_type'] == "protein":
                    for res in mol['residues']:
                        for atom in res['atoms']:
                            norm = (atom['relative_affinity'] - mean) / std
                            # if norm > 3:
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
    atomString += tempString + "  "

    tempString = str(atom['atom_name'])
    while (len(tempString) < 4):
        tempString += " "
    atomString += tempString

    tempString = str(res['res_name'])
    while (len(tempString) < 3):
        tempString = " " + tempString
    atomString += tempString + " "

    atomString += "A"

    tempString = str(res['res_id'])
    while (len(tempString) < 4):
        tempString = " " + tempString
    atomString += tempString

    atomString += "    "

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
    atomString += tempString + "  "

    atomString += "1.00"

    if atom['relative_affinity'] - int(atom['relative_affinity']) > 0:
        tempString = str(
            round(atom['relative_affinity'], 2))
        tempString += "0"
    else:
        tempString = str(
            round(atom['relative_affinity'], 2))
    while (len(tempString) < 5):
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

def thinGrid(pList, sphereGrid, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize, distFromProtein):

    for a in range(sphereGrid.shape[0]):
        for b in range(sphereGrid.shape[1]):
            for c in range(sphereGrid.shape[2]):
                nearP = False
                point = [sphereGrid[a, b, c], a, b, c]
                point = toPDBCoords(point, xOffset, yOffset, zOffset, minX, minY, minZ, bufferSize)
                for p in pList:
                    x = point[1] - p['x']
                    y = point[2] - p['y']
                    z = point[3] - p['z']
                    dist = sqrt((x*x)+(y*y)+(z*z))
                    if dist < distFromProtein:
                        nearP = True
                        break
                if nearP == False:
                    sphereGrid[a, b, c] = 0.0

    return sphereGrid

def visualizeProc(tasks_to_accomplish, tasks_that_are_done, jsonDir, gridDir, visualizationDir, hparams, bufferSize, intDist):

    model = CoolSystem(hparams).load_from_checkpoint(checkpoint_path='epoch=18.ckpt')
    model.cuda(0)
    model.eval()

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

                majorOrient = 0
                minorOrient = 0

                try:
                    grids = np.load(gridDir + filename + '-' + str(majorOrient) + '-' + str(minorOrient) + '.npz')
                except:
                    print("Could not find {}".format(filename))
                    continue
                atom_grid_np = grids['a']
                atom_grid_np = atom_grid_np[np.newaxis, :, :, :, :]
                mask_grid_np = grids['b']
                mask_grid_np = mask_grid_np[np.newaxis, :, :, :, :]

                with torch.no_grad():
                    atom_grid = torch.from_numpy(atom_grid_np).cuda()
                    mask_grid = torch.from_numpy(mask_grid_np).cuda()

                    y_hat = model.forward(atom_grid)
                    # loss = F.binary_cross_entropy_with_logits(y_hat,  mask_grid).detach().cpu().numpy()
                    y_hat = torch.sigmoid(y_hat)

                    targetGrid = y_hat.detach().cpu().numpy().squeeze()

                np.savez_compressed(os.path.join('data/predictions', filename), a=targetGrid.squeeze(), b=mask_grid_np.squeeze())
                     
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
    gridDir = "data/tmp-all/"

    visualize(jsonDir, gridDir, visualizationDir, hparams, bufferSize, intDist, num_workers=1)

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
