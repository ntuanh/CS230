import numpy as np
import time
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split

# from preprocessing import Preprocessing
from preprocessing_data import TriangleData

DATA_NOISE = 0.25
DATA_TEST = 0.2
np.random.seed(42)

# X, y = make_moons(n_samples=50000, noise=DATA_NOISE)


class DNN:
    def __init__(self, learning_rate, batch, verbose=False):
        """
        input data
        shape of X: dimension x size
        shape of y: dimension x size
       :param learning_rate:
       :param batch:
       :param verbose:
        """
        self.lr = learning_rate
        self.batch = batch
        self.verbose = verbose

        # preprocessing = Preprocessing()
        self.X, self.X_test, self.y, self.y_test = TriangleData(num_samples=50000, test_size=DATA_TEST).data()

        # self.X, self.X_test, self.y, self.y_test = train_test_split(
        #     X, y, test_size=0.2, random_state=42
        # )
        # self.y = self.y.reshape((1, len(self.y)))
        # self.y_test = self.y_test.reshape((1, len(self.y_test)))
        # self.X = self.X.T
        # self.X_test = self.X_test.T

        if self.verbose:
            print(f"len of self.y {len(self.y)}")
            print(f"len of self y[0] {len(self.y[0])}")
            print(f"len of X {len(self.X)}")
            print(f"len of X[0] {len(self.X[0])}")
            print(f"shape of X {self.X.shape}")
            print(f"shape of y {self.y.shape}")
            print(f"input size {len(self.X)}")

        self.layers = [len(self.X), 8,  4, 2,  1]
        self.act_funcs = ["None", "relu", "tanh", "sigmoid", "sigmoid"]
        self.L = len(self.layers) - 1

        self.speed_w = {}
        self.speed_b = {}

        self.momentum = 0.9

        self.w = {}
        self.b = {}
        self.z = {}
        self.a = {}

        # adam optimizer
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.t = 0
        self.mt = {}
        self.vt = {}
        self.epsilon = 1e-8

        # dropout
        self.p = 0     # fraction of dropout
        print(f"Dropout ratio {self.p}")
        self.dropped_nodes = {}     # storage lists index dropped node

        # display
        self.mean_acc = 0

    # [Activation functions] start
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
        return 2 * self.sigmoid(2*z) - 1

    def tanh_deriv(self, z):
        return 1 - self.tanh(z)*self.tanh(z)


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
            return self.tanh(z)
        else:
            return self.relu_deriv(z)

    # [ Activation functions ] end
    # [ Initialization ]
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

    def dropout(self , n):
        count = int(self.p * n)
        return np.random.choice(np.arange( n ), size=count, replace=False)

    def forward(self, x):
        self.a[0] = x  # (input, batch)

        for i in range(1, self.L + 1):
            self.z[i] = self.w[i].T @ self.a[i-1] + self.b[i]
            self.a[i] = self.activation_func(self.z[i], i)
            # print(f"shape of a[i] {self.a[i].shape}")
            rand_matrix = np.random.rand(*self.a[i].shape)
            mask = (rand_matrix < (1 - self.p))
            self.a[i] *= mask
        return self.a[self.L]

    def backward(self, y):
        m = y.shape[1]
        self.t += 1

        dz = self.a[self.L] - y   # (1, batch)

        for i in range(self.L, 0, -1):
            a_prev = self.a[i-1]

            dw = a_prev @ dz.T / m
            db = np.sum(dz, axis=1, keepdims=True) / m

            # momentum
            self.speed_w[i] = self.momentum * self.speed_w[i] + (1 - self.momentum) * dw
            self.speed_b[i] = self.momentum * self.speed_b[i] + (1 - self.momentum) * db

            # adam optimizer update
            self.w[i] -= self.adam_step(dw, f"w{i}")
            self.b[i] -= self.adam_step(db, f"b{i}")

            if i > 1:
                da_prev = self.w[i] @ dz
                # dz = da_prev * self.sigmoid_deriv(self.z[i-1])
                dz = da_prev * self.activation_func_deriv(self.z[i-1], i)

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

    def train(self, epochs=100):
        # print(f"shape of X: {self.X.shape}")
        # print(f"shape of Y: {self.y.shape}")
        start_time = time.time_ns()

        print(f"Training loop\t Data noise: {DATA_NOISE} \t test: {DATA_TEST}")
        # print("enable shuffle data ")
        self.init_weights()

        for e in range(epochs):
            # shuffle data
            m = self.X.shape[1]
            permutation = np.random.permutation(m)
            self.X = self.X[:, permutation]
            self.y = self.y[:, permutation]

            # self.lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + np.cos(e * np.pi/ epochs))
            for i in range(0, self.X.shape[1], self.batch):
                Xb = self.X[:, i:i+self.batch]
                Yb = self.y[:, i:i+self.batch]

                self.forward(Xb)
                cost = self.compute_cost(Yb)
                self.backward(Yb)

            if e % 10 == 0:
                acc = self.accuracy()
                self.mean_acc += acc
                print(f"Epoch {e} \t loss {cost:.7f} \t acc: {acc*100:.2f} %") #, learning rate {self.lr:.7f}")

        print(f"Finish with total time {(time.time_ns() - start_time) / 1e9} \t mean acc = {self.mean_acc*100 / 20}%")
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

model = DNN(learning_rate=0.001, batch=32, verbose=False)
model.train(epochs=200)
