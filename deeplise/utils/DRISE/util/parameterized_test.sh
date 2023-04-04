#!/bin/bash

#first parameter is simply directory of pdb files
#second parameter is molecule type for query

./bin/DRISE -i $1 -c file -m $2 -p train
./bin/DRISE -i data/complexJsons -c file -m $2 -p bsp
python3 util/GeneratePNHistograms.py -i data/complexJsons/ -o data/histograms/ -f residue -b 100
