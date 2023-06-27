from sklearn.metrics import mean_squared_error,r2_score

from utils import AverageMeter
from utils.optim_utils import get_current_lr

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import matplotlib.pyplot as plt
import cv2

import os

def train_model(epoch, batch_train, device, optimizer, model, criterion, lr_fn, args):
    model.train()

    ## Training
    true_labels = []
    pred_labels = []
    train_loss = AverageMeter()
    for i, (ecg,age,eid,weight) in enumerate(batch_train):
        # print(x_tr)
        # Zero grad
        model.zero_grad()

        ecg = ecg.float().to(device)
        age = age.float().to(device)
        weight = weight.float().to(device)
        
        pred = model(ecg)
        
        loss = criterion(age, pred,weight)

        loss.backward()
        optimizer.step()

        train_loss.update(loss.item(), len(ecg))
        true_labels.extend(list(age.cpu().numpy().astype(int)))
        pred_labels.extend(list(pred.detach().cpu().numpy().astype(int)))

        print(">>> Epoch [%3d/%3d] | Iter [%3d/%3d] | Loss %.6f" % (epoch+1, args.nb_epoch, i+1, len(batch_train), train_loss.avg), end='\r')

    # train performance
    mse = mean_squared_error(true_labels, pred_labels)
    print("\n>>> Training MSE : %.4f (LR %.7f)" % (mse, get_current_lr(optimizer)))
    
    return train_loss.avg, mse


def valid_model(epoch, batch_val, device, model, criterion, args):
    model.eval()

    val_loss = AverageMeter()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for j, (ecg,age,eid,weight) in enumerate(batch_val):
            ecg = ecg.float().to(device)
            age = age.float().to(device)
            weight = weight.float().to(device)

            pred = model(ecg)
            loss = criterion(age, pred,weight)

            val_loss.update(loss.item(), len(ecg))
            true_labels.extend(list(age.cpu().numpy().astype(int)))
            pred_labels.extend(list(pred.detach().cpu().numpy().astype(int)))

    # validation performance
    print(true_labels)
    print(pred_labels)
    mse = mean_squared_error(true_labels, pred_labels)
    print(">>> Validation MSE : %.4f" % mse)
    
    return val_loss.avg, mse



def test_model(batch_te, device, model, criterion, args):
    model.eval()

    test_loss = AverageMeter()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for j, (ecg,age,eid,weight) in enumerate(batch_te):
            ecg = ecg.float().to(device)
            age = age.float().to(device)
            weight = weight.float().to(device)

            pred = model(ecg)
            loss = criterion(age, pred,weight)

            test_loss.update(loss.item(), len(ecg))
            true_labels.extend(list(age.cpu().numpy().astype(int)))
            pred_labels.extend(list(pred.detach().cpu().numpy().astype(int)))

    # test performance
    mse = mean_squared_error(true_labels, pred_labels)
    print(">>> Test MSE : %.4f" % mse)
    
    # test performance
    r2 = r2_score(true_labels, pred_labels)
    print(">>> Test R-square : %.4f" % r2)

    return test_loss.avg, mse
