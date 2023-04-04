from pytorch_lightning import Trainer
from model import CoolSystem
from pytorch_lightning.loggers import TensorBoardLogger
import os
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from argparse import ArgumentParser
import torch.nn
from PIL import Image
import numpy as np
import torch
from utils.dataset import BasicDataset
from torchvision import transforms
from torchviz import make_dot, make_dot_from_trace

def main(hparams):
    # ------------------------
    # 1 INIT LIGHTNING MODEL
    # ------------------------
    model = CoolSystem(hparams).load_from_checkpoint(checkpoint_path='epoch=18.ckpt')

    # x = torch.randn(1, 20, 100, 100, 100).requires_grad_(True)
    # y = model(x)
    # make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)])).render()

    # model.init_weights()
    # model.configure_optimizers()

    # ------------------------
    # 2 INIT TRAINER
    # ------------------------
    logger = TensorBoardLogger("tb_logs", name="my_model", version="new-data-actualFix-0-new")
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=15,
        strict=False,
        verbose=False,
        mode='min'
    )

    numGpus = 1
    if hparams.num_tpu_cores != None:
        numGpus = 0

    trainer = Trainer(max_epochs=200, accumulate_grad_batches=1, 
                  gpus=numGpus, precision=32, logger=logger, 
                  tpu_cores=hparams.num_tpu_cores, resume_from_checkpoint='epoch=18.ckpt') #resume_from_checkpoint='epoch=12.ckpt'

    if hparams.num_tpu_cores == None:
        hparams.num_tpu_cores = 0

    # ------------------------
    # 3 START TRAINING
    # ------------------------

    # trainerLR = Trainer(gpus=numGpus, precision=32)

    # # Run learning rate finder
    # lr_finder = trainerLR.lr_find(model, min_lr=1e-5, max_lr=1.0, num_training=500)

    # # Results can be found in
    # lr_finder.results

    # # Plot with
    # lr_finder.plot(suggest = True, show = True)

    # # Pick point based on plot, or get suggestion
    # new_lr = lr_finder.suggestion()
    # print("Best LR: " + str(new_lr))

    # # update hparams of the model
    # model.hparams.lr = new_lr
    
    # trainer.fit(model)

    # ------------------------
    # 4 Test Model
    # ------------------------

    model.eval()
    trainer.test(model)

    # view tensorboard logs 
    print(f'View tensorboard logs by running\ntensorboard --logdir {os.getcwd()}/tb_logs')
    print('and going to http://localhost:6006 on your browser')

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="Ranger: learning rate")
    parser.add_argument("--image_scale", type=float, default=1.0, help="What size the images should be")
    parser.add_argument("--val_percent", type=float, default=0.1, help="What percentage of the dataset you should use to validate the model")
    parser.add_argument("--n_channels", type=int, default=19, help="Number of input channels. In this case, it should always be 3.")
    parser.add_argument("--n_classes", type=int, default=1, help="Number of output channels. In this case, it should also be 3.")
    parser.add_argument("--bilinear", type=bool, default=False, help="Determines if the model should use a learned upscaling or not. By default it chooses to learn.")
    parser.add_argument("--num_tpu_cores", type=int, default=None, help="Number of TPU cores to use. I have this as a holdover for when I used this skeleton on Google TPUs.")

    hparams = parser.parse_args()

    main(hparams)