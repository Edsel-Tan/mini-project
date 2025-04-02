from utils_edit import *
import multiprocessing
import random
import os
import numpy as np

os.sched_setaffinity(0, range(os.cpu_count()))

NUM_GAMES = 24000
NUM_PROCESSES = 16  # Changed from NUM_THREADS to NUM_PROCESSES

def generate_games(idx):
    np.random.seed(idx)
    data = set()
    for _ in range(NUM_GAMES // NUM_PROCESSES):
        current = State()
        while not current.is_terminal():
            random_action = current.get_random_valid_action()
            current = current.change_state(random_action)
            data.add(str(current))
    return data

if __name__ == "__main__":
    pool = multiprocessing.Pool(processes = NUM_PROCESSES)
    data = pool.map(generate_games, range(NUM_PROCESSES))

    while len(data) > 1:
        n = len(data)
        pool = multiprocessing.Pool(processes = n//2)
        data = pool.starmap(set.union, zip(data[:n//2], data[n//2:]))

    print("\n".join(data[0]))
