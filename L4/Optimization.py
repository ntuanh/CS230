import numpy as np

class Optimization:
    def __init__(self):
        pass
    def adam_step(self, gt, key):
        """
       :param gt: The gradient (dw or db)
       :param key: A unique string/tuple identifying the parameter (e.g., "w1", "b1")
        """
        # 1. Update the persistent state in the dictionaries
        self.mt[key] = self.beta_1 * self.mt[key] + (1 - self.beta_1) * gt
        self.vt[key] = self.beta_2 * self.vt[key] + (1 - self.beta_2) * (gt ** 2)

        # 2. Bias correction
        # Note: Use self.t (the global update counter)
        mt_corr = self.mt[key] / (1 - self.beta_1 ** self.t)
        vt_corr = self.vt[key] / (1 - self.beta_2 ** self.t)

        # 3. Calculate and return the update
        return self.lr * mt_corr / (np.sqrt(vt_corr) + self.epsilon)