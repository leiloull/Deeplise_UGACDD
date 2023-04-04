# DeepLise

DeepLise Source

## TODO
Add instructions for use.

## Setup

1. Install Anaconda Python Environment
2. Create Python environment for DeepLise
    1. conda create -n deeplise python=3 anaconda
    2. conda activate deeplise
    3. conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
    4. conda install biopython
    5. conda install -c conda-forge pytorch-lightning
    6. pip install echoAI torchviz
3. Install DRISE dependencies
    - README located in utils/DRISE/README.md
4. Create JSONs for DeepLISE
    1. cd utils/DRISE
    2. Convert PDB files in utils/DRISE/data/pdbFolder to JSON files
        - python util/convertPDBs.py
    3. Use DRISE to type atoms and determine surface
        1. make
            - Use -j argument followed by number of threads to spead up compilation
        2. ./bin/DRISE -i data/inputJsons -c file -m dna -p train
            - This operates on JSON files in data/inputJsons and outputs to data/complexJsons
        3. cp data/complexJsons/* ../../data/inputJsons/
5. Create grid files for DeepLISE
    1. cd data
    2. python preprocess.py
6. Create predictions
    - python createPredictions.py
    - outputs to data/predictions
7. Create output JSONs
    1. python createOutputJsons.py
        - outputs to data/outputJsons
    2. Visualize data
        1. python createVisualizations2.py
        2. Open VMD and load file in data/sites
8. Cluster scores
    1. python clusterScores.py
    2. Visualize data
        1. python createVisualizations2.py
        2. Open VMD and load file in data/sites-pred


    

