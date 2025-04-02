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

def f(s):
    s = s[15:].split()
    board = np.array([[[[0 for i in range(3)]for j in range(3)] for k in range(3)] for l in range(3)])
    x = s[0]
    w = int(s[2])
    d = int(s[3])
    l = int(s[4][:-1])
    fill_num = int(x[90])
    if fill_num == 2:
        w,l = l,w
    prev_local_action = int(x[91])
    if prev_local_action == 9:
        prev_local_action = None
    else:
        prev_local_action = (prev_local_action // 3, prev_local_action % 3)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    board[a][b][c][d] = int(x[a*27+b*9+c*3+d])
    return x,State(fill_num=fill_num, prev_local_action = prev_local_action, board=board),w,d,l

def g(i):
    data = {}
    if i < 10:
        i = "0" + str(i)
    with open(f"datagen/stage1-ncmts/depth{i}.txt", "r") as file:
        print(f"loading {i}", flush=True)
        d = file.readlines()
        d = d[:len(d)//16]
        dr = map(f, d)

    for x,j,w,d,l in dr:
        if x not in data:
            data[x] = [j,0,0,0]
        data[x][1] += w
        data[x][2] += d
        data[x][3] += l

    for x in data:
        j,w,d,l = data[x]
        if w + l == 0:
            v = 0
        else:
            v = ((w / (w + l)) * 2 - 1)
        data[x] = (j, v)

    print(f"{i} done.", flush=True)
    return data

class BoardDataset(Dataset):

    def __init__(self):
        data = {}
        pool = multiprocessing.Pool(processes = 81)
        datas = pool.map(g, range(81))

        print(f"Parallel done.", flush=True)
        self.data = []
        for data in datas:
            self.data.extend(data.values())
        print(f"Init done.", flush=True)
                
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        state = self.data[idx][0]
        value = torch.tensor(self.data[idx][1] if self.data[idx][0].fill_num == 1 else -self.data[idx][1], dtype=torch.float32).unsqueeze(0)
        
        return StateToTensor(state), value
        

                

        