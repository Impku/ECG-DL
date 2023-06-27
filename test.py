import cv2
import random
import os, sys
from glob import glob
import numpy as np

from model import load_model
from model.core import train_model, valid_model, test_model

from utils.data_loader import load_dataloader
from utils.config import TestParserArguments
from utils.optim_utils import load_optimizer, load_loss_function, CosineWarmupLR

import torch
import torch.nn as nn
import warnings


warnings.filterwarnings(action='ignore')

torch.backends.cudnn.benchmark = True

# Seed
RANDOM_SEED = 1234
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

if __name__ == '__main__':
    # Argument
    args = TestParserArguments()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    criterion = load_loss_function(args).to(device)

    test_loader = load_dataloader(args)

    # Model
    model = load_model(args).to(device)

    test_loss, test_f1 = test_model(test_loader, device, model, criterion, args)
