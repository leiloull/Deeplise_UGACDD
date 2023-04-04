from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, grids_dir):
        self.grids_dir = grids_dir

        self.ids = [splitext(file)[0] for file in listdir(grids_dir)
                    if not file.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        idx = self.ids[i]
        grid_file = glob(self.grids_dir + idx + '*')

        assert len(grid_file) == 1, \
            f'Either no image or multiple images found for the ID {idx}: {grid_file}'

        grids = np.load(grid_file[0])

        atom_grid = grids['a']
        oneHot_grid = grids['b']

        torchImage = torch.from_numpy(atom_grid)
        torchOneHot = torch.from_numpy(oneHot_grid)

        return {'image': torchImage, 'target': torchOneHot}
