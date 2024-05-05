# generating PBPs that preferentially bind to one plastic surface over another
import numpy as np
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_dir, '../camsol_calculation/'))
from camsol_calculation import CamSol_calc


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.total_reward = 0.0
        self.visits = 0
        self.action = action

    def expand(self, actions):
        for action in actions:
            if action not in self.children:
                new_state = self.state + [action]
                self.children[action] = Node(new_state, parent=self, action=action)
         
    def ucb(self, action, exploration_param):
        if self.children[action].visits == 0:
            return float('inf')
        else:
            return (self.children[action].total_reward / self.children[action].visits +
                    exploration_param * np.sqrt(2 * np.log(self.visits) / self.children[action].visits))

    def best_child(self, exploration_param):
        return max(self.children, key=lambda action: self.ucb(action, exploration_param))

    def num2str(self, state):
        aa_dic = {
            0: "A", 1: "D", 2: "E", 3: "F", 4: "G", 5: "H",
            6: "I", 7: "K", 8: "L", 9: "M", 10: "N", 11: "Q",
            12: "R", 13: "S", 14: "T", 15: "V", 16: "W", 17: "Y"
        }
        return ''.join(aa_dic[num] for num in state)


    def rollout(self, surrogate_model_1, surrogate_model_2, max_length, hsf):
        current_state = self.state.copy()
        while len(current_state) < max_length:
            action = np.random.choice(18)
            current_state.append(action)
        return current_state, -np.squeeze(surrogate_model_1.predict(np.asarray(current_state).reshape(-1, 12)))\
        + np.squeeze(surrogate_model_2.predict(np.asarray(current_state).reshape(-1, 12))) + hsf * CamSol_calc(self.num2str(current_state))

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


def mcts(root, surrogate_model_1, surrogate_model_2, num_iterations, exploration_param=1.0, hsf=1.0):
    # surrogate_model_1 is the model for the plastic that is preferred
    best_seq, best_score = [], 0.0
    for _ in range(num_iterations):
        node = root
        while node.children:
            action = node.best_child(exploration_param)
            node = node.children[action]
        if len(node.state) < 12:
            node.expand(range(18))
        seq, reward = node.rollout(surrogate_model_1, surrogate_model_2, 12, hsf)
        node.backpropagate(reward)
       
        if best_score >= -reward:
            best_seq = seq
            best_score = -reward
            
    return best_seq, best_score