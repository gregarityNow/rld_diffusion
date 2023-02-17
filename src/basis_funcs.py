
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
from tqdm import tqdm

import pathlib

dataLoc = "/tmp/f002nb9/"
pathlib.Path(dataLoc).mkdir(parents=True,exist_ok=True);

def getSuffix(class_emb_dim, w, epoch = -1, step = -1):
    ret = ""
    if epoch > -1:
        ret += "epoch"+str(epoch) + "_"

    if step > -1:
        ret += "step"+str(epoch) + "_"

    if class_emb_dim is not None:
        ret += "classEmb"+str(class_emb_dim) + "_"
    else:
        ret += "classEmbNone_"
    return ret

def getOutPath(suffix = ""):
    return "/users/Etu2/21210942/Documents/rld/out/" + suffix + "/"

from torch.utils.data import Dataset
class MnistDataset(Dataset):
    def __init__(self, ds):
        self.images = torch.from_numpy(ds.images.data()).unsqueeze(1) / 255.0
        self.labels = ds.labels.data()
        self.channels = 1
        self.image_shape = self.images[0, 0].shape

    def __getitem__(self, index):
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.images)

from datamaestro import prepare_dataset
def get_train_data(quickie):
    ds = prepare_dataset("com.lecun.mnist")
    train_data = MnistDataset(ds.train)
    if quickie:
        train_data.images = train_data.images[:100]
        train_data.labels = train_data.labels[:100]

    return train_data


from typing import Optional
from functools import partial
import os
import pickle
import math
def dumpRes(res):
    outPath = getOutPath()

    resPath = outPath + "/" + "results.pickle"
    if os.path.exists(resPath):
        with open(resPath,"rb") as fp:
            oldRes = pickle.load(fp)
        res = list(set(res + oldRes));
    with open(resPath,"wb") as fp:
        pickle.dump(res, fp);
    print("dumped to",outPath);


def prepare_folders(reset):
    outPath = getOutPath();
    if reset:
        import shutil
        shutil.rmtree(outPath);
    pathlib.Path(getOutPath("img")).mkdir(parents=True, exist_ok=True);
