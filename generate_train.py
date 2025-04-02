from utils_edit import *
import multiprocessing
import random
import os

os.sched_setaffinity(0, range(os.cpu_count()))

NUM_NODES = 50
NUM_GAMES = 100
THRESHOLD = 50
NUM_PROCESSES = 16
# NUM_NODES = 1
# NUM_GAMES = 10
# THRESHOLD = 5
# NUM_PROCESSES = 16
data = [dict() for _ in range(NUM_PROCESSES)]

def generate_train(states, idx):
    data = {}
    np.random.seed(idx)
    for n,s in enumerate(states):
        if n % 10 == 0:
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
        for _ in range(NUM_GAMES):
            q = [s]
            current = s
            while not current.is_terminal():
                action = current.get_random_valid_action()
                current = current.change_state(action)
                q.append(current)
            outcome = current.terminal_utility()
            if outcome == 1:
                o = 0 #Win
            elif outcome == 0:
                o = 1 #Lose
            else:
                o = 2 #Draw
            for state in q:
                if state not in data:
                    data[state] = [0,0,0]
                data[state][o] += 1
    return data

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
    data = pool.starmap(generate_train, zip([random.sample(d, NUM_NODES) for i in range(NUM_PROCESSES)], range(NUM_PROCESSES)))

    while len(data) > 1:
        n = len(data)
        print(f"Collating {len(data)}", flush=True)
        pool = multiprocessing.Pool(processes = n//2)
        data = pool.starmap(collate, zip(data[:n//2], data[n//2:]))
    
    collected_data = data[0]
    towrite = []
    for s in collected_data:
        if sum(collected_data[s]) > THRESHOLD:
            towrite.append(str(s) + " " + str(collected_data[s]))

    with open("datagen/train.out", "w+") as file:
        file.write("\n".join(towrite))

    print("Done!")