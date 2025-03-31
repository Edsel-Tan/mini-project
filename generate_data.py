from utils_edit import *
from threading import Thread, Lock
import random

NUM_GAMES = 10**6

data = set()
mutex = Lock()

def generate_games():
    global data
    for _ in range(NUM_GAMES):
        current = State()
        while not current.is_terminal():
            random_action = current.get_random_valid_action()
            current = current.change_state(random_action)
            with mutex:
                data.add(str(current))

if __name__ == "__main__":
    threads = [Thread(target = generate_games, args = ()) for _ in range(8)]
    for t in threads:
        t.start()

    for t in threads:
        t.join()

    print(len(data))
    for i in data:
        print(i)

    

    