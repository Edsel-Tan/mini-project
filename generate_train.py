from utils_edit import *
import multiprocessing
import random
import os

os.sched_setaffinity(0, range(os.cpu_count()))

NUM_NODES = 20000
NUM_GAMES = 50
THRESHOLD = 0
NUM_PROCESSES = 16
data = [dict() for _ in range(NUM_PROCESSES)]

class Node:
    def __init__(self, state):
        self.state = state
        self.actions = state.get_all_valid_actions()
        self.children = {}

    def get_random_valid_action(self):
        return self.actions[np.random.randint(len(self.actions))]

    def move(self, action):
        if action not in self.children:
            self.children[action] = Node(self.state.change_state(action))
        return self.children[action]

def generate_train(states, idx):
    data = {}
    np.random.seed(idx)
    for n,s in enumerate(states):
        if n % 1000 == 0:
            print(f"Thread {idx} has completed {n} nodes.", flush=True)
        fill_num = int(s[0])
        prev_local_action = (int(s[1]), int(s[2]))
        if prev_local_action == (3,3):
            prev_local_action = None
        board = np.array([[[[0 for i in range(3)]for j in range(3)] for k in range(3)] for l in range(3)])
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        board[i][j][k][l] = int(s[3+l+k*3+j*9+i*27])
        s = State(board=board, fill_num=fill_num, prev_local_action=prev_local_action)
        data[s] = [0,0,0]
        for _ in range(NUM_GAMES):
            root = Node(s)
            current = root
            while not current.state.is_terminal():
                action = current.get_random_valid_action()
                current = current.move(action)
            outcome = current.state.terminal_utility()
            if outcome == 1:
                o = 0 #Win
            elif outcome == 0:
                o = 1 #Lose
            else:
                o = 2 #Draw
            data[s][o] += 1
    return "\n".join([str(s) + " " + str(data[s]) for s in data])

def collate(dic1, dic2):
    dic = {}
    for i in dic1:
        if i not in dic:
            dic[i] = [0,0,0]
        for j in range(3):
            dic[i][j] += dic1[i][j]
    for i in dic2:
        if i not in dic:
            dic[i] = [0,0,0]
        for j in range(3):
            dic[i][j] += dic2[i][j]
    return dic


if __name__ == "__main__":
    random.seed(42)
    with open("datagen/data.out", 'r') as file:
        d = file.readlines()
    
    pool = multiprocessing.Pool(processes = NUM_PROCESSES)
    samples = random.sample(d, NUM_NODES * NUM_PROCESSES)
    data = pool.starmap(generate_train, zip([samples[i*NUM_NODES:(i+1)*NUM_NODES] for i in range(NUM_PROCESSES)], range(NUM_PROCESSES)))

    with open("datagen/train.out", "w+") as file:
        file.write("\n".join(data))

    print("Done!")