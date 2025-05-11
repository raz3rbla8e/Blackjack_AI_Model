from typing import Tuple, List, Any
import numpy as np
from dataclasses import dataclass

#Stores information for one transition
@dataclass
class Transition:

    state: Tuple
    action: int
    reward: float
    next_state: Tuple
    terminal: bool

#implements Prioritized Experience Replay (PER) from https://arxiv.org/abs/1511.05952
#he idea is to sample important transitions more often based on their TD error.
class PrioritizedReplayBuffer:

    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        self.capacity = capacity
        self.alpha = alpha  # priority exponent
        self.beta = beta    # importance sampling exponent
        self.beta_increment = 0.001
        self.transitions: List[Transition] = []
        self.priorities: List[float] = []
        self.position = 0
        self.priority_epsilon = 1e-6

    #add a transition to the buffer. Uses max priority if no error given.
    def add(self, transition: Transition, error: float = None) -> None:
        priority = self._get_max_priority() if error is None else (abs(error) + self.priority_epsilon) ** self.alpha
        
        if len(self.transitions) < self.capacity:
            self.transitions.append(transition)
            self.priorities.append(priority)
        else:
            self.transitions[self.position] = transition
            self.priorities[self.position] = priority
            
        self.position = (self.position + 1) % self.capacity
    """
    Sample transitions based on their priorities. Returns:
    - transitions: the sampled transitions
    - indices: their positions in the buffer
    - weights: importance sampling weights to correct for bias
    """
    def sample(self, batch_size: int) -> Tuple[List[Transition], List[int], np.ndarray]:
        if len(self.transitions) == 0:
            return [], [], np.array([])

        n_samples = min(batch_size, len(self.transitions))
        
        probs = np.array(self.priorities)
        probs = probs / sum(probs)
        
        indices = np.random.choice(len(self.transitions), n_samples, p=probs)
        
        weights = (len(self.transitions) * probs[indices]) ** (-self.beta)
        weights = weights / weights.max()
        
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        samples = [self.transitions[idx] for idx in indices]
        return samples, indices, weights
    #updates priorities based on new TD errors.
    def update_priorities(self, indices: List[int], errors: List[float]) -> None:
        for idx, error in zip(indices, errors):
            self.priorities[idx] = (abs(error) + self.priority_epsilon) ** self.alpha
    #returns max priority in buffer or 1.0 if empty.
    def _get_max_priority(self) -> float:
        return max(self.priorities) if self.priorities else 1.0
    #returns length of buffer in transitions
    def __len__(self) -> int:
        return len(self.transitions)