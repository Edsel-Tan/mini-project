from utils import State, Action
from submission import StudentAgent

class StudentAgent():
    def choose_action(self, state: State) -> Action:
        return state.get_all_valid_actions()[0]