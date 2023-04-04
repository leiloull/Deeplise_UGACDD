#!/bin/bash

#SBATCH --job-name=miq
#SBATCH --partition=paulxie_p
#SBATCH --time=168:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --gres=gpu:1
#SBATCH --mem=10gb
#SBATCH --mail-user=ll38965@uga.edu
#SBATCH --mail-type=ALL
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

cd /home/ll38965/deeplise

module load Anaconda3/2021.05
conda init deeplise
module load CUDA/11.2.1-GCC-8.3.0
ml FreeSASA/2.0.3-GCC-8.3.0
ml libxml2/2.9.9-GCCcore-8.3.0
ml json-c/0.15-GCCcore-8.3.0
export LDFLAGS=" -L /apps/eb/libxml2/2.9.9-GCCcore-8.3.0/lib -L /apps/eb/json-c/0.15-GCCcore-8.3.0/lib "
cd data
python preprocess.py
cd ..
python createPredictions.py
python createOutputJsons.py
python createVisualizations2.py
#python clusterScores.py
