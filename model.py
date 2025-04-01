import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import load_data

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

class NN(nn.module):
    def __init__(self):
        self.conv = nn.Sequential(
            nn.Conv2d(5, 64, kernel_size = 3, padding = 1, bias = True),
            nn.ELU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(128),
            nn.ELU(),
            nn.Conv2d(128, 64, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ELU(),
            nn.Conv2d(64, 1, kernel_size = 1, padding = 0, bias=False),
            nn.Linear(81, 1),
            nn.Tanh(),
        )

    def __forward__(self, x):
        return self.conv(x)

class BoardDataset(Dataset):

    def __init__(self):
        self.data = load_data()
    
    def __len__(self):
        #TODO
        raise NotImplementedError

    def __getitem__(self, idx):
        #TODO
        raise NotImplementedError
    