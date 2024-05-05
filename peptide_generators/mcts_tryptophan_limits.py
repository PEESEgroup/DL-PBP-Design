# generating PBPs with 12 amino acids with the "three-tryptophan constraint"
import numpy as np


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
            if self.state.count(16) < 3 or action != 16:
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

    def rollout(self, surrogate_model, max_length):
        current_state = self.state.copy()
        while len(current_state) < max_length:
            if current_state.count(16) < 3:
                action = np.random.choice(18)
            else:
                possible_actions = list(range(18))
                possible_actions.remove(16)
                action = np.random.choice(possible_actions)

            current_state.append(action)
        return current_state, -np.squeeze(surrogate_model.predict(np.asarray(current_state).reshape(-1, 12)))

    def backpropagate(self, reward):
        self.visits += 1
        self.total_reward += reward
        if self.parent:
            self.parent.backpropagate(reward)


def mcts(root, surrogate_model, num_iterations, exploration_param=1.0):
    best_seq, best_score = [], 0.0
    for _ in range(num_iterations):
        node = root
        while node.children:
            action = node.best_child(exploration_param)
            node = node.children[action]
        if len(node.state) < 12:
            node.expand(range(18))
        seq, reward = node.rollout(surrogate_model, 12)
        node.backpropagate(reward)
       
        if best_score >= -reward:
            best_seq = seq
            best_score = -reward
            
    return best_seq, best_score