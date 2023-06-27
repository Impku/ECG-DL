import cv2
import random
import os, sys
from glob import glob
import numpy as np

from model import load_model
from model.core import train_model, valid_model

from utils.data_loader import load_dataloader
from utils.config import ParserArguments
from utils.optim_utils import load_optimizer, load_loss_function

import torch
import torch.nn as nn
import warnings

import matplotlib.pyplot as plt


warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

# Seed
RANDOM_SEED = 45233
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


if __name__ == '__main__':
    # Argument
    args = ParserArguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Model
    model = load_model(args).to(device)

    # Loss
    criterion = load_loss_function(args).to(device)

    # Optimizer
    optimizer = load_optimizer(model, args)

    if args.mode == 'train':  ## for train mode
        print('Training start ...')
        train_loader, val_loader = load_dataloader(args)

        lr_fn = None
        # #####   Training and Validation loop   #####
        best_mse = 1000000
        best_loss = 100

        lossdict = {"train":[],"val":[]}
        msedict = {"train":[],"val":[]}

        for epoch in range(args.nb_epoch):
            train_loss, train_mse = train_model(epoch, train_loader, device, optimizer, model, criterion, lr_fn, args)
            val_loss, val_mse = valid_model(epoch, val_loader, device, model, criterion, args)
            lossdict["train"].append(train_loss)
            lossdict["val"].append(val_loss)
            msedict["train"].append(train_mse)
            msedict["val"].append(val_mse)
            if val_mse < best_mse:
                best_mse = val_mse

                torch.save(model.state_dict(),
                            os.path.join(args.exp, 'epoch_%03d_val_loss_%.4f_val_mse_%.4f.pth'%(epoch, val_loss, val_mse)))
            
                print("\r>>> Best score updated : F1 %.4f in %d epoch.\n" % (val_mse, epoch+1))
            else:
                print("\n")


            # loss plot
            plt.plot([i+1 for i in range(len(lossdict["train"]))],lossdict["train"])
            plt.savefig("plots/train_loss.png")
            plt.clf()

            plt.plot([i+1 for i in range(len(lossdict["val"]))],lossdict["val"])
            plt.savefig("plots/validation_loss.png")
            plt.clf()
        

            # loss plot
            plt.plot([i+1 for i in range(len(msedict["train"]))],msedict["train"])
            plt.savefig("plots/train_mse.png")
            plt.clf()

            plt.plot([i+1 for i in range(len(msedict["val"]))],msedict["val"])
            plt.savefig("plots/validation_mse.png")
            plt.clf()

        torch.save(model.state_dict(),
                            os.path.join(args.exp, 'epoch_%03d_val_loss_%.4f_val_mse_%.4f.pth'%(epoch, val_loss, val_mse)))

 
        