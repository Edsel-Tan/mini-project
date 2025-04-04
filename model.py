import torch
from torch import nn
from torch.utils.data import Dataset
import numpy as np
from utils import load_data, State, board_status, get_local_board_status
import multiprocessing

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

NUM_FEATURES = 4

def StateToTensor(state : State):
    board_tensor = np.zeros((NUM_FEATURES, 9, 9,))
    player = state.fill_num
    local_board = get_local_board_status(state.board)
    
    if state.prev_local_action and local_board[state.prev_local_action[0]][state.prev_local_action[1]] == 0:
        x, y = state.prev_local_action
        board_tensor[0][x*3:(x+1)*3,y*3:(y+1)*3] = (state.board[x][y] == 0)
        
    else:
        for i in range(3):
            for j in range(3):
                if local_board[i][j] == 0:
                    board_tensor[0][i*3:(i+1)*3,j*3:(j+1)*3] = (state.board[i][j] == 0)

    for i in range(3):
        for j in range(3):
            board_tensor[1][i*3:(i+1)*3,j*3:(j+1)*3] = (state.board[i][j] == player)
            board_tensor[2][i*3:(i+1)*3,j*3:(j+1)*3] = (state.board[i][j] == 3-player)

    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    board_tensor[3][a*3+c][b*3+d] = local_board[c][d]

    return torch.tensor(board_tensor, dtype=torch.float32)

class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(NUM_FEATURES, 32, kernel_size = 3, padding = 1, bias = True),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size = 3, padding = 1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            # nn.Conv2d(64, 64, kernel_size = 3, padding = 1, bias=True),
            # nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size = 3, padding = 1, bias=True),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size = 1, padding = 0, bias=True),
            nn.Flatten(),
            nn.Linear(81, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.conv(x)

class BoardDataset(Dataset):

    def __init__(self):
        self.data = load_data()
                
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][0]
        value = torch.tensor(self.data[idx][1] if self.data[idx][0].fill_num == 1 else -self.data[idx][1], dtype=torch.float32).unsqueeze(0)
        
        return StateToTensor(state), value
        

                

        