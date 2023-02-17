
import torch
from torch import nn
device = torch.device("cuda" if torch.cuda.is_available else "cpu")
from tqdm import tqdm

import pathlib

##flags to set!
dataLoc = "/tmp/f002nb9/"
baseOutPath = "/users/Etu2/21210942/Documents/rld/out/"

pathlib.Path(dataLoc).mkdir(parents=True,exist_ok=True);
pathlib.Path(baseOutPath).mkdir(parents=True,exist_ok=True);

def getSuffix(class_emb_dim, w, epoch = -1,version=0, dsName = "MNIST", schedType = "linear"):
    ret = dsName + "_"
    if epoch > -1:
        ret += "epoch"+str(epoch) + "_"


    if class_emb_dim is not None:
        ret += "classEmb"+str(class_emb_dim) + "_"
    else:
        ret += "classEmbNone_"

    ret += schedType + "_"
    ret += "w"+str(w)+"_"

    ret += "version"+str(version)



    return ret

def getOutPath(suffix = ""):
    return baseOutPath + suffix + "/"

from torch.utils.data import Dataset
class CustomDataset(Dataset):
    def __init__(self, ds):
        self.images = torch.from_numpy(ds.images.data()).unsqueeze(1) / 255.0
        self.labels = ds.labels.data()
        self.channels = 1
        self.image_shape = self.images[0, 0].shape
    def __getitem__(self, index):
        return self.images[index], self.labels[index]
    def __len__(self):
        return len(self.images)




from typing import Optional
from functools import partial
import os
import pickle
import math
def dumpRes(d):
    outPath = getOutPath()

    resPath = outPath + "/" + "results.pickle"
    if os.path.exists(resPath):
        with open(resPath,"rb") as fp:
            oldRes = pickle.load(fp)
        oldRes.append(d)
    else:
        oldRes = [d]
    with open(resPath,"wb") as fp:
        pickle.dump(oldRes, fp);
    print("dumped to",outPath);


def prepare_folders(reset):
    outPath = getOutPath();
    if reset:
        import shutil
        shutil.rmtree(outPath);
    pathlib.Path(getOutPath("img")).mkdir(parents=True, exist_ok=True);
