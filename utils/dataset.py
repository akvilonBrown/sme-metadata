import torch
from torch.utils import data
import numpy as np

class CustomDataSet(data.Dataset):
    def __init__(self,
                 inputs: np.ndarray,
                 targets: np.ndarray,
                 metalabels: list,
                 mmap: dict, 
                 transform=None                                 
                 ):
        self.inputs = inputs
        self.targets = targets
        self.metalabels = metalabels
        self.transform = transform
        self.mmap = mmap
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long
        self.metalabels_dtype = torch.long
        self.meta_dtype = torch.float32

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self,
                    index: int):

        # Load input and target
        x, m, y = self.inputs[index], self.metalabels[index], self.targets[index]

        # Preprocessing
        if self.transform is not None:
            x, y = self.transform(x, y)
        
        m = self.mmap[m]

        # Typecasting
        x, m, y = torch.from_numpy(x).type(self.inputs_dtype), \
                  torch.from_numpy(m).type(self.meta_dtype), \
                  torch.from_numpy(y).type(self.targets_dtype).squeeze()

        return x, m, y
