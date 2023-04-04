import numpy as np
import matplotlib.pyplot as plt

dataCSV = 'extendedTypes.csv'

def makeTriangleOccurenceHistogram():

    csv = np.genfromtxt(dataCSV, delimiter=",")
    csv[0, 0] = 8

    totalTriObserved = 0

    for row in csv:
        totalTriObserved += row[0]

    print(totalTriObserved)

    print('Allocating expanded array')

    expandedValues = np.empty(int(totalTriObserved), dtype=np.float32)

    print('Populating expanded array')

    index = 0

    for row in csv:
        expandedValues[index:(index+int(row[0]))] = row[1]
        index += int(row[0])
    
    print('Making Histogram')

    _ = plt.hist(expandedValues, bins=100)
    plt.title("Histogram with 'auto' bins")
    print('Showing Histogram')
    plt.show()


def makeTriangleTypeHistogram():

    csv = np.genfromtxt(dataCSV, delimiter=",")
    csv[0, 0] = 8

    totalTriObserved = 0

    for row in csv:
        totalTriObserved += 1

    print(totalTriObserved)

    print('Allocating array')

    expandedValues = np.empty(int(totalTriObserved), dtype=np.float32)

    print('Populating array')

    index = 0

    for row in csv:
        expandedValues[index] = row[1]
        index += 1

    print('Making Histogram')

    _ = plt.hist(expandedValues, bins=100)
    plt.title("Histogram with 'auto' bins")
    print('Showing Histogram')
    plt.show()

#makeTriangleTypeHistogram()

makeTriangleOccurenceHistogram()
