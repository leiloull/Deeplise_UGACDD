#!/bin/bash

mkdir data/predJsons
rm -r data/complexJsons
cp -r ../DRISE_Scores/DRISE_aAT_AVG/complexJsons data/
python  util/clusterScores.py > DRISE_aAT_AVG.txt
mv data/predJsons data/predJsons_DRISE_aAT_AVG

mkdir data/predJsons
rm -r data/complexJsons
cp -r ../DRISE_Scores/DRISE_aAT_Max/complexJsons data/
python  util/clusterScores.py > DRISE_aAT_Max.txt
mv data/predJsons data/predJsons_DRISE_aAT_Max

mkdir data/predJsons
rm -r data/complexJsons
cp -r ../DRISE_Scores/DRISE_aAT_SUM/complexJsons data/
python  util/clusterScores.py > DRISE_aAT_SUM.txt
mv data/predJsons data/predJsons_DRISE_aAT_SUM

mkdir data/predJsons
rm -r data/complexJsons
cp -r ../DRISE_Scores/DRISE_sAT_AVG/complexJsons data/
python  util/clusterScores.py > DRISE_sAT_AVG.txt
mv data/predJsons data/predJsons_DRISE_sAT_AVG

mkdir data/predJsons
rm -r data/complexJsons
cp -r ../DRISE_Scores/DRISE_sAT_Max/complexJsons data/
python  util/clusterScores.py > DRISE_sAT_Max.txt
mv data/predJsons data/predJsons_DRISE_sAT_Max

mkdir data/predJsons
rm -r data/complexJsons
cp -r ../DRISE_Scores/DRISE_sAT_SUM/complexJsons data/
python  util/clusterScores.py > DRISE_sAT_SUM.txt
mv data/predJsons data/predJsons_DRISE_sAT_SUM