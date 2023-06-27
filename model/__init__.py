import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from .resnet import ResNet1d


def load_model(args):
    #####   Model   #####
    if 'resnet' in args.network:
        model = ResNet1d(input_dim=(12, 4096),
                     blocks_dim=list(zip([64, 128, 196, 256, 320], [4096, 1024, 256, 64, 16])),
                     n_classes=1,
                     kernel_size=17,
                     dropout_rate=0.8)
        
    else:
        raise ValueError("resnet and efficientnet are only supported now.")

    if args.resume:
        model.load_state_dict(torch.load(args.resume))


    return model