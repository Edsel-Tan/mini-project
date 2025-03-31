from utils_edit import *
from threading import Thread
import random

NUM_GAMES = 24000
NUM_THREADS = 24

data = [set() for _ in range(NUM_THREADS)]

def generate_games(idx : int):
    global data
    for _ in range(NUM_GAMES // NUM_THREADS):
        current = State()
        while not current.is_terminal():
            random_action = current.get_random_valid_action()
            current = current.change_state(random_action)
            data[idx].add(str(current))

if __name__ == "__main__":
    threads = [Thread(target = generate_games, args = (idx,)) for idx in range(NUM_THREADS)]
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    total = set()
    for i in data:
        total = total.union(i)
    
    for i in total:
        print(i)

    

    