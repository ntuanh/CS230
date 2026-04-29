import numpy as np

class ActivationFunctions:
    def __init__(self):
        pass

    def sigmoid(self, z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    def tanh(self, z):
        return 2 * self.sigmoid(2 * z) - 1

    def tanh_deriv(self, z):
        return 1 - self.tanh(z) * self.tanh(z)

    def activation_func(self, z, i):
        if self.act_funcs[i] == "sigmoid":
            return self.sigmoid(z)
        elif self.act_funcs[i] == "tanh":
            return self.tanh(z)
        else:
            return self.relu(z)

    def activation_func_deriv(self, z, i):
        if self.act_funcs[i] == "sigmoid":
            return self.sigmoid_deriv(z)
        elif self.act_funcs[i] == "tanh":
            return self.tanh_deriv(z)
        else:
            return self.relu_deriv(z)
