#!/bin/bash
make clean
make -j9
./bin/DRISE -i data/inputJsons/ -c file -m dna -p train
./bin/DRISE -i data/complexJsons -c file -m dna -p bsp
python3 util/GeneratePNHistograms.py -i data/complexJsons/ -o data/histograms -f residue -b 40
