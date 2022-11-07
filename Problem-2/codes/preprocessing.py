import os
import numpy as np
from torch.utils.data import Dataset, DataLoader

class eICUDataSet(Dataset):
    def __init__(self, config):
        self.root_file = config["root_file"]
        self.in_arr = np.array([i[0] for i in np.load(self.root_file).flatten()])
        self.frac = config["frac"]
        self.label = config["label"]
        self.transform = config["transform"]
        self.target_transform = config["target_transform"]
        self.maxlength = config["maxlength"]
        self.length = round(self.frac*len(self.in_arr))
        self.xunique = np.unique(self.in_arr)
        self.Integral = len(self.in_arr)
        self.dim = (len(self.xunique), 1)
    def __len__(self):
        return self.maxlength

    def __getitem__(self, idx):
        arr = np.random.choice(self.in_arr, self.length, replace = False)
        arr = np.array([np.count_nonzero(arr == i) for i in self.xunique])
        #arr = arr/np.sum(arr) #dx = 1
        label = self.label
        self.arrlength = len(arr)
        if self.transform:
            image = self.transform(arr)
        if self.target_transform:
            label = self.target_transform(label)
        return arr, label

#custom_dataset = eICUDataLoader("/content/eICU_age.npy", 0.25, 10000)

def MakeTrainOrTestData(dataset: Dataset, config: dict):
    return DataLoader(dataset, config["batch_size"], config["shuffle"])
