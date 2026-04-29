import numpy as np

class Initialization:
    def __init__(self):
        pass
    def xavier_init(self, n_in, n_out):
        std = np.sqrt(2.0 / (n_in + n_out))
        return np.random.normal(loc=0.0, scale=std, size=(n_in,n_out))

    def he_init(self, n_in, n_out):
        std = np.sqrt(2.0 / n_in)
        return np.random.normal(loc=0.0, scale=std, size=(n_in,n_out))

    def init_weights(self):
        for i in range(1, self.L + 1):
            if self.act_funcs[i] == "relu":
                self.w[i] = self.he_init(self.layers[i-1], self.layers[i])
            else :
                self.w[i] = self.xavier_init(self.layers[i-1], self.layers[i])
            self.b[i] = np.zeros((self.layers[i], 1))
            self.speed_b[i] = np.zeros((self.layers[i], 1))
            self.speed_w[i] = np.zeros((self.layers[i - 1], self.layers[i])) * np.sqrt(1 / self.layers[i - 1])

            self.mt[f"w{i}"] = np.zeros_like(self.w[i])
            self.vt[f"w{i}"] = np.zeros_like(self.w[i])

            # Initialize mt and vt for Biases
            self.mt[f"b{i}"] = np.zeros_like(self.b[i])
            self.vt[f"b{i}"] = np.zeros_like(self.b[i])