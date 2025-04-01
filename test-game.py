import torch
from utils import *
import time
from torch import tensor
from torch import nn
from collections import OrderedDict
import numpy as np

# from dummy import StudentAgent
from submission import StudentAgent
from submission_2 import StudentAgent2

def run(your_agent: StudentAgent, random_agent: StudentAgent, start_num: int):
    timeout_count = 0
    invalid_count = 0
    state = State(fill_num=start_num)
    turn_count = 0
    actions = []
    while not state.is_terminal():
        turn_count += 1
        print(f"Starting turn {turn_count}")
        random_action = state.get_random_valid_action()
        if state.fill_num == 2:
            start_time = time.time()
            action = random_agent.choose_action(state.clone())
            end_time = time.time()
            if end_time - start_time > 3:
                print("Other agent time out!")
                timeout_count += 1
                action = random_action
        else:
            start_time = time.time()
            action = your_agent.choose_action(state.clone())
            end_time = time.time()
            if end_time - start_time > 3:
                print("Your agent time out!")
                timeout_count += 1
                action = random_action
        if not state.is_valid_action(action):
            assert state.fill_num == 1
            print("Your agent made an invalid action!")
            invalid_count += 1
            action = random_action
        state = state.change_state(action)
        actions.append(action)
    if state.terminal_utility() == 1:
        print("You win!")
    elif state.terminal_utility() == 0:
        print("You lose!")
    else:
        print("Draw")
    print(f"Timeout count: {timeout_count}")
    print(f"Invalid count: {invalid_count}")
    print(action[0])

your_agent = StudentAgent(2)
random_agent = StudentAgent(4)

run(your_agent, random_agent, 1)
run(your_agent, random_agent, 2)