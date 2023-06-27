import os
import cv2
import random
from collections import defaultdict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader

import SimpleITK as sitk

from glob import glob

import pickle
import gzip


def compute_weights(ages, max_weight=np.inf):
    _, inverse, counts = np.unique(ages, return_inverse=True, return_counts=True)
    weights = 1 / counts[inverse]
    normalized_weights = weights / sum(weights)
    w = len(ages) * normalized_weights
    # Truncate weights to a maximum
    if max_weight < np.inf:
        w = np.minimum(w, max_weight)
        w = len(ages) * w / sum(w)
    return w

class ECGDataset(Dataset):
    def __init__(self, is_Train, args):
        self.args = args
        self.is_Train = is_Train
        self.ecglist,self.agelist,self.idlist,self.weightlist = self._load_image_list()

        print("# of %s images : %d" % ('training' if is_Train else 'validation', len(self.ecglist)))

    def __getitem__(self, index):

        ecg_path = self.ecglist[index]
        age = self.agelist[index]
        eid = self.idlist[index]
        weight = self.weightlist[index]
    
        # Load Image
        ecg = pd.read_csv(ecg_path).values.T[:,:4096]
        # print(ecg.shape)
        return ecg,age,eid,weight

    def __len__(self):
        return len(self.ecglist)


    def _load_image_list(self):
        
        impath = sorted(glob(os.path.join(self.args.data_root,'*.csv')))[:]
        df = pd.read_csv("../EKG_waveform_date_label.tsv","\t")
        df["weights"] = compute_weights(df["age"])
        ecglist = []
        agelist = []
        idlist = []
        weightlist = []
        for path in impath:
            with open(path,"r") as f:
                if len(f.read().split("\n")) < 3000:
                    continue
            ecg_ID = os.path.basename(path).split(".")[0]
            try:
                age = df[df.waveID == ecg_ID]["age"].values[0]
                weight = df[df.waveID == ecg_ID]["weights"].values[0]
            except:
                print(ecg_ID)
                continue

            ecglist.append(path)
            agelist.append(age)
            idlist.append(ecg_ID)
            weightlist.append(weight)

        length = len(ecglist)
        if self.is_Train:
            ecglist = ecglist[:int(length*0.6)]
            agelist = agelist[:int(length*0.6)]
            idlist = idlist[:int(length*0.6)]
            weightlist = weightlist[:int(length*0.6)]
        elif self.args.mode == "train":
            ecglist = ecglist[int(length*0.6):int(length*0.8)]
            agelist = agelist[int(length*0.6):int(length*0.8)]
            idlist = idlist[int(length*0.6):int(length*0.8)]
            weightlist = weightlist[int(length*0.6):int(length*0.8)]
        else:
            ecglist = ecglist[int(length*0.8):]
            agelist = agelist[int(length*0.8):]
            idlist = idlist[int(length*0.8):]
            weightlist = weightlist[int(length*0.8):]


        return ecglist,agelist,idlist,weightlist


def load_dataloader(args):
    if args.mode == "train":
        tr_set = ECGDataset(is_Train=True, args=args)
        val_set = ECGDataset(is_Train=False, args=args)

        batch_train = DataLoader(tr_set, batch_size=args.batch_size, shuffle=True, num_workers=28, pin_memory=True)
        batch_val = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=28, pin_memory=True)

        return batch_train, batch_val
    else:
        test_set = ECGDataset(is_Train=False, args=args)
        batch_test = DataLoader(test_set, batch_size=args.batch_size, shuffle=True, num_workers=28, pin_memory=True)        

        return batch_test