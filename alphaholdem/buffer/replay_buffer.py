# buffer/replay_buffer.py
import numpy as np
import torch

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.buffer = []

    def push(self, state, action, log_prob, value, reward, done, next_state):
        self.buffer.append((state, action, log_prob, value, reward, done, next_state))
        if len(self.buffer) > self.capacity:
            self.buffer.pop(0)

    def sample(self, batch_size):
        idxs = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in idxs]
        return zip(*batch)

    def clear(self):
        self.buffer = []
