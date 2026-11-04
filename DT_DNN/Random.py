import numpy as np

class Random(object):
    def __init__(self, a_dim):
        self.a_dim = a_dim
        self.pointer = 0
        self.update_cnt = 0

    def choose_action(self, s):
        return np.random.uniform(0.0, 1.0, size=(1, self.a_dim))

    def initialize_target_networks(self):
        pass

    def store_transition(self, s, a, r, s_):
        pass

    def learn(self):
        pass

   