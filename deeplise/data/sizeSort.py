import numpy as np
import os
import csv

gridDir = "unsorted-grids"

gridDict = {}

for filename in os.listdir(gridDir):
    grids = np.load(gridDir + '/' + filename)
    size = grids['a'].flatten().shape[0] + grids['b'].flatten().shape[0]
    gridDict[filename] = size

sortedList = sorted(gridDict, key=gridDict.__getitem__)

count = 0
for g in sortedList:
    print(str(count) + ': ' + g)
    count += 1

with open('sortedSizes.csv', 'w', newline='') as myfile:
    wr = csv.writer(myfile)
    wr.writerow(sortedList)

# val = input("Enter val start value: ")

# for filename in os.listdir(gridDir):
#     grids = np.load(gridDir + '/' + filename)
#     size = grids['a'].flatten().shape[0] + grids['b'].flatten().shape[0]
#     gridDict[filename] = size