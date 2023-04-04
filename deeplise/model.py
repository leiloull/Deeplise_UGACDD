import os
import multiprocessing

import torch
import torch.nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms, datasets

import pytorch_lightning as pl

from utils.unet_parts import *
from utils.dataset import BasicDataset
from utils.ranger import Ranger
from collections import OrderedDict

# from utils.vizualize_data import showResults

def startWeights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)
    elif type(m) == nn.ConvTranspose2d:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.00)

class UNET(nn.Module):

    def __init__(self, hparams):
        super(UNET, self).__init__()

        baseFilters = 32

        self.n_channels = 19
        self.n_classes = 1
        self.bilinear = False

        self.pre = nn.Conv3d(self.n_channels, self.n_channels, kernel_size=9, padding=4)
        self.inc = ConvBlock(self.n_channels, baseFilters)
        self.down1 = Down(baseFilters, baseFilters*2)
        self.down2 = Down(baseFilters*2, baseFilters*4)
        self.down3 = Down(baseFilters*4, baseFilters*4)
        self.up1 = Up(baseFilters*8, baseFilters*2, self.bilinear)
        self.up2 = Up(baseFilters*4, baseFilters, self.bilinear)
        self.up3 = Up(baseFilters*2, baseFilters, self.bilinear)
        self.outc = OutConv(baseFilters, self.n_classes)

    def forward(self, x):
        x = self.pre(x)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits

class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()

        print(hparams)

        self._hparams = hparams

        self.dir_grids = 'data/grids/'
        self.dif_val_grids = 'data/val-grids/'
        self.dir_checkpoint = 'checkpoints/'
        self.val_percent = 0.1
        # self.batch_size = hparams.batch_size
        self.num_tpu_cores = 0

        self.dataset = BasicDataset(self.dir_grids)
        self.val_dataset = BasicDataset(self.dif_val_grids)
        self.test_dataset = BasicDataset(self.dir_grids)

        # self.n_val = int(len(self.dataset) * self.val_percent)
        # self.n_train = len(self.dataset) - self.n_val
        # train, val = random_split(self.dataset, [self.n_train, self.n_val])
        self.train_set = self.dataset
        self.val_set = self.val_dataset

        self.n_channels = 19
        self.n_classes = 1
        self.bilinear = False

        self.model = UNET(hparams)
        # self.model = FCRES(hparams)

    def init_weights(self):
        self.model = self.model.apply(startWeights)

    def forward(self, x):
        logits = self.model(x)
        return logits

    def training_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['target']

        y_hat = self.forward(x)

        loss = F.binary_cross_entropy_with_logits(y_hat,  y)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['target']
        y_hat = self.forward(x)
        # print(y_hat)
        return {'val_loss': F.binary_cross_entropy_with_logits(y_hat,  y)}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'val_loss': avg_loss}
        return {'avg_val_loss': avg_loss, 'log': tensorboard_logs}
        
    def test_step(self, batch, batch_idx):
        x = batch['image']
        y = batch['target']
        y_hat = self.forward(x)
        y0_hat = torch.zeros(y_hat.shape, dtype=torch.float32, device=torch.device('cuda:0'))

        return {'test_loss': F.binary_cross_entropy_with_logits(y_hat,  y),
                'zero_loss': F.binary_cross_entropy_with_logits(y0_hat,  y)}

    def test_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg0_loss = torch.stack([x['zero_loss'] for x in outputs]).mean()
        tensorboard_logs = {'test_loss': avg_loss,
                            'zero_loss': avg0_loss}
        return {'avg_test_loss': avg_loss, 'avg_zero_loss': avg0_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = Ranger(self.parameters(), lr=0.0001)
        # optimizer = apex.optimizers.FusedSGD(self.parameters(), lr=self.hparams.lr, momentum=0.25, dampening=0, weight_decay=0, 
        #                                     nesterov=True, wd_after_momentum=False, materialize_master_grads=True)
        # optimizer = torch.optim.LBFGS(self.parameters(), lr=self.hparams.lr, max_iter=20, max_eval=None, tolerance_grad=1e-07, 
        #                             tolerance_change=1e-09, history_size=100, line_search_fn=None)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, cooldown=1, threshold=0.01)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 200, eta_min=0, last_epoch=-1)
        return [optimizer] #, [scheduler]

    def train_dataloader(self):
        
        dataset = self.train_set

        # required for TPU support
        sampler = None
        # if (self.num_tpu_cores != None) or (self.num_tpu_cores != 0):
        #     sampler = torch.utils.data.distributed.DistributedSampler(
        #         dataset,
        #         num_replicas=xm.xrt_world_size(),
        #         rank=xm.get_ordinal(),
        #         shuffle=True
        #     )
        # else:
        sampler = torch.utils.data.RandomSampler(dataset, replacement=False, num_samples=None)

        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(), 
            pin_memory=True
        )

        return loader

    def val_dataloader(self):
        dataset = self.val_set

        # required for TPU support
        sampler = None
        # if (self.num_tpu_cores != None) or (self.num_tpu_cores != 0):
        #     sampler = torch.utils.data.distributed.DistributedSampler(
        #         dataset,
        #         num_replicas=xm.xrt_world_size(),
        #         rank=xm.get_ordinal(),
        #         shuffle=True
        #     )

        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=self.hparams.batch_size,
            num_workers=multiprocessing.cpu_count(), 
            pin_memory=True
        )

        return loader

    def test_dataloader(self):
        dataset = self.test_dataset

        # required for TPU support
        sampler = None
        # if (self.num_tpu_cores != None) or (self.num_tpu_cores != 0):
        #     sampler = torch.utils.data.distributed.DistributedSampler(
        #         dataset,
        #         num_replicas=xm.xrt_world_size(),
        #         rank=xm.get_ordinal(),
        #         shuffle=True
        #     )

        loader = DataLoader(
            dataset,
            sampler=sampler,
            batch_size=1,
            num_workers=multiprocessing.cpu_count(), 
            pin_memory=True
        )

        return loader