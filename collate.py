import multiprocessing
from utils import State
import os
os.sched_setaffinity(0, range(os.cpu_count()))
import numpy as np

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
    with open(f"datagen/stage1-mcts/depth{i}.txt", "r") as file:
        print(f"loading {i}", flush=True)
        d = file.readlines()
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
            v = ((w / (w + l)) * 2 - 1) * (d / (d + w + l))
        data[x] = (j, v)

    print(f"{i} done.", flush=True)
    return data

data = {}
pool = multiprocessing.Pool(processes = 81)
datas = pool.map(g, range(81))

print(f"Parallel done.", flush=True)
DATA = []
for data in datas:
    DATA.extend(data.values())
print(f"Init done.", flush=True)

import pickle
with open("datagen/data.pkl", "wb+") as file:
    pickle.dump(DATA, file)