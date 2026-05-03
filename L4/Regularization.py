import numpy as np

class Regularization:
    def __init__(self):
        pass
    def dropout(self , n):
        count = int(self.p * n)
        return np.random.choice(np.arange( n ), size=count, replace=False)