from math import pi, cos

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import _LRScheduler

def get_current_lr(optimizer):
    return optimizer.state_dict()['param_groups'][0]['lr']

def load_optimizer(model, args):
    if not args.bias_decay:
        weight_params = []
        bias_params = []
        for n, p in model.named_parameters():
            if 'bias' in n:
                bias_params.append(p)
            else:
                weight_params.append(p)
        parameters = [{'params' : bias_params, 'weight_decay' : 0},
                      {'params' : weight_params}]
    else:
        parameters = model.parameters()

    if args.optim.lower() == 'rmsprop':
        optimizer = torch.optim.RMSprop(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(parameters, lr=args.learning_rate, momentum=args.momentum, weight_decay=args.wd)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.learning_rate)
  
    return optimizer


class compute_loss(nn.Module):

    def __init__(self):
        super(compute_loss, self).__init__()

    def forward(self,ages, pred_ages, weights):
        diff = ages.flatten() - pred_ages.flatten()
        loss = torch.sum(weights.flatten() * diff * diff)
        return loss


def load_loss_function(args):
    criterion = compute_loss()

    return criterion



