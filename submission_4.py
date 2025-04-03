import torch
from utils import *
import time
from torch import tensor
from torch import nn
from collections import OrderedDict
import random

class StudentAgent:

    def __init__(self, depth = 5):
        """Instantiates your agent.
        """
        random.seed(42)
        self.depth = depth
        pass

    def choose_action(self, state: State) -> Action:
        """Returns a valid action to be played on the board.
        Assuming that you are filling in the board with number 1.

        Parameters
        ---------------
        state: The board to make a move on.
        """
        t, value, action = self.minimax(state, -float('inf'), float('inf'), self.depth)
        return action
    
    def check_local_win(board, player):
      """Check if a 3x3 board is won by a player."""
      for row in range(3):
          if all(board[row, col] == player for col in range(3)):
              return True
      for col in range(3):
          if all(board[row, col] == player for row in range(3)):
              return True
      if all(board[i, i] == player for i in range(3)) or all(board[i, 2 - i] == player for i in range(3)):
          return True
      return False
    
    def heuristic(state : State):
      board = state.board
      score = 0
      player = 1
      opponent = 3 - player
      
      for global_r in range(3):
        for global_c in range(3):
            local_board = board[global_r, global_c]  # Extract the 3x3 local board

            # Winning and losing checks
            if StudentAgent.check_local_win(local_board, player):
                score += 100  # Winning a local board is strong
            elif StudentAgent.check_local_win(local_board, opponent):
                score -= 100  # Opponent winning is bad

            # Center control
            if local_board[1, 1] == player:
                score += 10
            elif local_board[1, 1] == opponent:
                score -= 10

            # Corner control
            for r, c in [(0, 0), (0, 2), (2, 0), (2, 2)]:
                if local_board[r, c] == player:
                    score += 5
                elif local_board[r, c] == opponent:
                    score -= 5

      return score + len(state.get_all_valid_actions()) * (1 if state._state.fill_num == 1 else -1)
        
    
    def minimax(self, state: State, alpha : float, beta : float, depth : int, time_left : float = 2.85):
        start_time = time.time()
        isMax = state.fill_num == 1
        m = 1 if isMax else -1

        if state.is_terminal():
            return time.time() - start_time, (state.terminal_utility() - 0.5) * 2, None
        if time_left < 0:
            return time.time() - start_time, StudentAgent.heuristic(state), None
        if depth == 0:
            value = StudentAgent.heuristic(state)
            end_time = time.time()
            return end_time - start_time, value, None
        
        if isMax:
            best_value = -float('inf')
            best_action = None
            for action in state.get_all_valid_actions():
                t, value, _ = self.minimax(state.change_state(action), alpha, beta, depth - 1, time_left - (time.time() - start_time))
                if value > best_value:
                    best_value = value
                    best_action = action
                if value > beta:
                    break
                alpha = max(value, alpha)
            return time.time() - start_time, best_value, best_action

        else:
            best_value = float('inf')
            best_action = None
            for action in state.get_all_valid_actions():
                t, value, _ = self.minimax(state.change_state(action), alpha, beta, depth - 1, time_left - (time.time() - start_time))
                if value < best_value:
                    best_value = value
                    best_action = action
                if value < alpha:
                    break
                beta = min(value, beta)
            return time.time() - start_time, best_value, best_action
