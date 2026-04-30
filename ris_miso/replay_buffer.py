import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, max_size: int = 100_000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.states      = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.actions     = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_states = np.zeros((max_size, state_dim),  dtype=np.float32)
        self.rewards     = np.zeros((max_size, 1),          dtype=np.float32)
        self.not_dones   = np.zeros((max_size, 1),          dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.next_states[self.ptr] = next_state
        self.rewards[self.ptr]     = reward
        self.not_dones[self.ptr]   = 1.0 - float(done)
        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        to = lambda x: torch.FloatTensor(x[idx]).to(self.device)
        return to(self.states), to(self.actions), to(self.next_states), \
               to(self.rewards), to(self.not_dones)
