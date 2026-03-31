import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from preprocessing import Preprocessing

np.random.seed(42)

# X, Y = make_moons(n_samples=30000, noise=0.2)

class DNN:
    def __init__(self, learning_rate, batch):
        self.lr = learning_rate
        self.batch = batch

        preprocessing = Preprocessing()
        self.X , self.X_test , self.y , self.y_test = preprocessing.data()

        self.layers = [preprocessing.get_input_size() , 128 ,  32 ,  1]
        self.L = len(self.layers) - 1

        self.speed_w = {}
        self.speed_b = {}

        self.momentum = 0.9

        self.w = {}
        self.b = {}
        self.z = {}
        self.a = {}

    # [ Activation functions ] start
    def sigmoid(self, z):
        return np.where(z >= 0,
                        1 / (1 + np.exp(-z)),
                        np.exp(z) / (1 + np.exp(z)))

    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    def relu(self , z):
        return np.maximum(0, z)

    def relu_deriv(self, z):
        return (z > 0).astype(float)

    # [ Activation functions ] end

    def init_weights(self):
        for i in range(1, self.L + 1):
            self.w[i] = np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(1/self.layers[i-1])
            self.b[i] = np.zeros((self.layers[i], 1))
            self.speed_b[i] = np.zeros((self.layers[i], 1))
            self.speed_w[i] = np.zeros((self.layers[i - 1], self.layers[i])) * np.sqrt(1 / self.layers[i - 1])

    def forward(self, X):
        self.a[0] = X  # (input, batch)

        for i in range(1, self.L + 1):
            self.z[i] = self.w[i].T @ self.a[i-1] + self.b[i]
            self.a[i] = self.sigmoid(self.z[i])
            # self.a[i] = self.relu(self.z[i])
        return self.a[self.L]

    def backward(self, Y):
        m = Y.shape[1]

        dz = self.a[self.L] - Y   # (1, batch)

        for i in range(self.L, 0, -1):
            a_prev = self.a[i-1]

            dw = a_prev @ dz.T / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            # momentum
            self.speed_w[i] = self.momentum * self.speed_w[i] + ( 1 - self.momentum) * dw
            self.speed_b[i] = self.momentum * self.speed_b[i] + ( 1 - self.momentum) * db


            # update
            self.w[i] -= self.lr * self.speed_w[i]
            self.b[i] -= self.lr * self.speed_b[i]

            if i > 1:
                da_prev = self.w[i] @ dz
                dz = da_prev * self.sigmoid_deriv(self.z[i-1])

    def train(self, epochs=100):
        # print(f"shape of X : {self.X.shape}")
        # print(f"shape of Y : {self.y.shape}")

        lr_min = 0.0001
        lr_max = 0.1

        print("Training loop")
        self.init_weights()

        for e in range(epochs):
            self.lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(e * np.pi/ epochs))
            for i in range(0, self.X.shape[1], self.batch):
                Xb = self.X[:, i:i+self.batch]
                Yb = self.y[:, i:i+self.batch]

                self.forward(Xb)
                cost = self.compute_cost(Yb)
                self.backward(Yb)

            if e % 10 == 0:
                print(f"Epoch {e} loss {cost:.7f} , acc: {self.accuracy():.3f} , learning rate {self.lr:.7f}")

            if cost < 0.001 :
                self.lr = 0.001

        pred = self.forward(self.X_test)
        print("pred mean:", np.mean(pred))
        print("pred min/max:", pred.min(), pred.max())

    def train_accuracy(self):
        pred = self.forward(self.X)
        pred = (pred > 0.5).astype(int)
        return np.mean(pred == self.y)

    def accuracy(self):
        pred = self.forward(self.X_test)
        pred = (pred > 0.5).astype(int)
        return np.mean(pred == self.y_test)

    def compute_cost(self, Y):
        A = self.a[self.L]
        m = Y.shape[1]

        A = np.clip(A, 1e-8, 1 - 1e-8)

        return -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m




model = DNN(learning_rate=0.01, batch=32)
model.train(epochs=200)