import numpy as np
import time

DATA_SIZE = 5000
DATA_NOISE = 0.25
DATA_TEST = 0.2
np.random.seed(42)

# use data of sklearn
# from sklearn.datasets import make_moons
# from sklearn.model_selection import train_test_split
# X, y = make_moons(n_samples=DATA_SIZE, noise=DATA_NOISE)

# use my data - Triangle Data
# from preprocessing import Preprocessing
from preprocessing_data import TriangleData
from Activation_funcs import ActivationFunctions
from Initialization import Initialization
from Evaluation import Evaluation
from Regularization import Regularization
from Optimization import Optimization




class DNN(ActivationFunctions , Initialization , Evaluation , Regularization , Optimization):
    def __init__(self, learning_rate, batch, verbose=False , my_data=True ):
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

        self.my_data = my_data
        if my_data :
            print("Triangle Data !")
            self.X, self.X_test, self.y, self.y_test = TriangleData(num_samples=DATA_SIZE, test_size=DATA_TEST).data()
        # else :
        #     print("Make Moon Data !")
        #     self.X, self.X_test, self.y, self.y_test = train_test_split(
        #         X, y, test_size=0.2, random_state=42
        #     )
        #     self.y = self.y.reshape((1, len(self.y)))
        #     self.y_test = self.y_test.reshape((1, len(self.y_test)))
        #     self.X = self.X.T
        #     self.X_test = self.X_test.T

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
        self.lambd = 0.01

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
        self.p = 0.015    # fraction of dropout
        print(f"Dropout ratio {self.p}")
        self.dropped_nodes = {}     # storage lists index dropped node

        # display
        self.mean_acc = 0

        self.grads = {}

    def forward(self, x , check_gradient = False ):
        self.a[0] = x  # (input, batch)

        for i in range(1, self.L + 1):
            self.z[i] = self.w[i].T @ self.a[i-1] + self.b[i]
            self.a[i] = self.activation_func(self.z[i], i)
            if check_gradient is False:
                rand_matrix = np.random.rand(*self.a[i].shape)
                mask = (rand_matrix < (1 - self.p))
                self.a[i] = (self.a[i] * mask) / (1 - self.p)
        return self.a[self.L]

    def backward(self, y , update = True):
        m = y.shape[1]
        self.t += 1

        dz = self.a[self.L] - y   # (1, batch)

        for i in range(self.L, 0, -1):
            a_prev = self.a[i-1]

            dw = a_prev @ dz.T / m
            # L2 regularization
            dw += (self.lambd / m ) * self.w[i]
            db = np.sum(dz, axis=1, keepdims=True) / m

            # momentum
            self.speed_w[i] = self.momentum * self.speed_w[i] + (1 - self.momentum) * dw
            self.speed_b[i] = self.momentum * self.speed_b[i] + (1 - self.momentum) * db

            # SAVE gradients for checking
            self.grads[f"dw{i}"] = dw
            self.grads[f"db{i}"] = db

            if update:
                # adam optimizer update
                self.w[i] -= self.adam_step(dw, f"w{i}")
                self.b[i] -= self.adam_step(db, f"b{i}")

            if i > 1:
                da_prev = self.w[i] @ dz
                dz = da_prev * self.activation_func_deriv(self.z[i-1], i - 1)

    def train(self, epochs=100):
        # print(f"shape of X: {self.X.shape}")
        # print(f"shape of Y: {self.y.shape}")
        start_time = time.time_ns()

        print(f"Training loop\t Data noise: {DATA_NOISE} \t test: {DATA_TEST}")
        self.init_weights()

        for e in range(epochs):
            # shuffle data
            m = self.X.shape[1]
            permutation = np.random.permutation(m)
            self.X = self.X[:, permutation]
            self.y = self.y[:, permutation]

            for i in range(0, self.X.shape[1], self.batch):
                Xb = self.X[:, i:i+self.batch]
                Yb = self.y[:, i:i+self.batch]

                self.forward(Xb)
                cost = self.compute_cost(Yb)
                self.backward(Yb)
                # print(f"shape of Yb {Yb.shape}")    # (1 , 32) 1 feature , batch size 32

            if e % 10 == 0:
                acc = self.accuracy()
                if e > 0 :
                    self.mean_acc += acc
                print(f"Epoch {e} \t loss {cost:.7f} \t acc: {acc*100:.2f} %") #, learning rate {self.lr:.7f}")

        total_time = (time.time_ns() - start_time) / 1e9
        print(f"Finish with total time {total_time} \t mean acc = {self.mean_acc*100 / 19:.3f}%")
        return total_time

model = DNN(learning_rate=0.001, batch=32, verbose=False , my_data=True)
model.init_weights()
X_check = model.X[:, :2]
Y_check = model.y[:, :2]
model.run_gradient_check(X_check, Y_check)
model.train(epochs=200)
