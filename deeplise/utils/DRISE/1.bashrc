#!/bin/bash
#SBATCH --job-name=makeenv
#SBATCH --partition=paulxie_p
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=40gb
#SBATCH --time=12:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
#SBATCH --mail-user=ll38965@uga.edu
#SBATCH --mail-type=ALL

#cd home/ll38965/deeplise/utils/DRISE/

module load Anaconda3/2021.05
activate deeplise
module load CUDA/11.2.1-GCC-8.3.0
ml FreeSASA/2.0.3-GCC-8.3.0
ml libxml2/2.9.9-GCCcore-8.3.0
ml json-c/0.15-GCCcore-8.3.0
export LDFLAGS=" -L /apps/eb/libxml2/2.9.9-GCCcore-8.3.0/lib -L /apps/eb/json-c/0.15-GCCcore-8.3.0/lib "

make clean

make
./bin/DRISE -i data/inputJsons -c file -m dna -p train
cp data/complexJsons/* ../../data/inputJsons/
