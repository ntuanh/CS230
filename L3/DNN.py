import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

np.random.seed(42)

X, Y = make_moons(n_samples=30000, noise=0.2)

class DNN:
    def __init__(self, learning_rate, batch):
        self.lr = learning_rate
        self.batch = batch

        # split
        X_train, X_test, y_train, y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42
        )

        # IMPORTANT: reshape to (features, batch)
        self.X = X_train.T
        self.y = y_train.reshape(1, -1)
        self.X_test = X_test.T
        self.y_test = y_test.reshape(1, -1)

        self.layers = [2, 4, 6 , 4 , 1]
        self.L = len(self.layers) - 1

        self.w = {}
        self.b = {}
        self.z = {}
        self.a = {}

    # [ Activation functions ] start
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_deriv(self, z):
        s = self.sigmoid(z)
        return s * (1 - s)

    # [ Activation functions ] end

    def init_weights(self):
        for i in range(1, self.L + 1):
            self.w[i] = np.random.randn(self.layers[i-1], self.layers[i]) * np.sqrt(1/self.layers[i-1])
            self.b[i] = np.zeros((self.layers[i], 1))

    def forward(self, X):
        self.a[0] = X  # (input, batch)

        for i in range(1, self.L + 1):
            self.z[i] = self.w[i].T @ self.a[i-1] + self.b[i]
            self.a[i] = self.sigmoid(self.z[i])

        return self.a[self.L]

    def backward(self, Y):
        m = Y.shape[1]

        dz = self.a[self.L] - Y   # (1, batch)

        for i in range(self.L, 0, -1):
            a_prev = self.a[i-1]

            dw = a_prev @ dz.T / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            # update
            self.w[i] -= self.lr * dw
            self.b[i] -= self.lr * db

            if i > 1:
                da_prev = self.w[i] @ dz
                dz = da_prev * self.sigmoid_deriv(self.z[i-1])

    def train(self, epochs=100):
        self.init_weights()

        for e in range(epochs):
            for i in range(0, self.X.shape[1], self.batch):
                Xb = self.X[:, i:i+self.batch]
                Yb = self.y[:, i:i+self.batch]

                self.forward(Xb)
                self.backward(Yb)

            if e % 10 == 0:
                print(f"Epoch {e}, acc: {self.accuracy():.3f}")

    def accuracy(self):
        pred = self.forward(self.X_test)
        pred = (pred > 0.5).astype(int)
        return np.mean(pred == self.y_test)



model = DNN(learning_rate=0.1, batch=20)
model.train(epochs=100)