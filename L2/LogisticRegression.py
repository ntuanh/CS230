import numpy as np
import pandas as pd
from build_data import Build

# simple data
x_train = [
    [0.1 , 0.2],
    [0.2 , 0.3]
]
y_train = [
    1 ,
    0
]

X , y = Build(10000).read_data()

class LogisticRegression:
    def __init__(self):
        self.X_temp = np.array(X)
        self.X = self.X_temp.reshape(10000 , 3, 1)
        self.y = np.array(y)
        self.w = {
            1 : np.random.rand(3 , 6) ,
            2 : np.random.rand(6 , 1)
        }
        self.b1 = np.random.rand(6 , 1)
        self.b = {
            1 : self.b1,
            2 : np.zeros((1,1))
        }

        self.depth = 1
        self.batch = 10  # batch index from 0
        self.learning_rate = 0.1

        self.pred = []
        # pass

    def load_data(self , verbose = True):
        wine_red = pd.read_csv('datasets/winequality-red.csv')
        wine_white = pd.read_csv('datasets/winequality-white.csv')

        if verbose :
            print(f'check wine red {wine_red.head()}')
            print(f'check wine white \n{wine_white.head()}')

    # Forward
    def compute_z(self , x , w , b):
        return np.dot(np.transpose(w) , x) + b

    def sigmoid(self , z):
        return 1 / (1 + np.exp(-z))

    def compute_a(self , z):
        return self.sigmoid(z)

    def forward(self , x):
        a = x
        for i in range(1 , self.depth + 2):
            z = self.compute_z(a , self.w[i] , self.b[i])
            a = self.compute_a(z)
        return a

    def loss(self):
        l = 0
        self.pred = []
        for i , x in enumerate(self.X):
            # x = np.transpose(x)
            p = self.forward(x)
            p = np.clip(p, 1e-8, 1 - 1e-8)
            self.pred.append(p[0][0])
            l += (self.y[i]*np.log(p) + (1-self.y[i])*np.log(1-p))
        l =  -l / len(self.X)
        return l[0][0]

    # Backward
    def d_sigmoid(self , z):    # derivative of sigmoid function
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def derivative_Loss_by_a(self , truth_value , prediction):
        return -truth_value / prediction + (truth_value -1 )/ (prediction - 1)

    def derivative_a_by_z(self , z):
        return self.d_sigmoid(z)

    def derivative_w2(self , x , y_truth):
        """
        compute derivative anf update weights of layer 2
        :param x: output of past layer
        :param y_truth: truth values
        :return: no return
        """
        z1 = np.transpose(self.w[1])@x + self.b[1]
        a1 = self.sigmoid(z1)
        z2 = np.transpose(self.w[2])@a1 + self.b[2]
        a2 = self.sigmoid(z2)

        dL_dz2 = a2 - y_truth

        dw2 = a1 @ dL_dz2.T
        return dw2

    def derivative_w1(self , x , y_truth):
        z1 = self.w[1].T @ x + self.b[1]
        a1 = self.sigmoid(z1)

        z2 = self.w[2].T @ a1 + self.b[2]
        a2 = self.sigmoid(z2)

        dZ2 = a2 - y_truth
        dZ1 = (self.w[2] @ dZ2) * (a1 * (1 - a1))

        dw1 = x @ dZ1.T
        return dw1

    def derivative_b1(self, x, y_truth):
        z1 = self.compute_z(x, self.w[1], self.b[1])
        a1 = self.compute_a(z1)
        z2 = self.compute_z(a1, self.w[2], self.b[2])
        a2 = self.compute_a(z2)
        dL_dz2 = self.derivative_Loss_by_a(y_truth, a2) * self.derivative_a_by_z(z2)

        dz2_da1 = self.w[2]
        dz2_da1 = np.transpose(dz2_da1)

        da1_dz1 = np.diag((a1*(1-a1)).flatten())

        db = dL_dz2 @ dz2_da1 @ da1_dz1
        db = np.transpose(db)
        return db

    def run(self , a , b, c , verbose = True ):
        print('start')
        print(self.X.shape)
        epoch = 200
        dw1 = 0
        dw2 = 0
        db1 = 0
        for iterator in range(epoch):
            if verbose and iterator % 50 == 0:
                print(iterator , self.loss())
            for i in range(len(self.X)):
                x = self.X[i]
                y_truth = self.y[i]
                if i % self.batch == 0 and i != 0 :
                    self.w[1] -= self.learning_rate * dw1 / self.batch
                    self.w[2] -= self.learning_rate * dw2 / self.batch
                    self.b[1] -= self.learning_rate * db1 / self.batch
                else:
                    dw1 += self.derivative_w1(x, y_truth)
                    dw2 += self.derivative_w2(x, y_truth)
                    db1 += self.derivative_b1(x, y_truth)

                if i % self.batch == 0:
                    dw1 = 0
                    dw2 = 0
                    db1 = 0


        self.test_set()
        print(f'accuracy {self.accuracy()}')
        if self.accuracy() > 90 :
            self.save_parameters()

        input_data = np.array([[a],[b],[c]])
        # print(input_data.shape)
        res = self.forward(input_data)
        print(res)
        if res > 0.5 :
            print('True')
        else :
            print('False')

    def test_set(self):
        for i in range(5):
            pred = self.forward(self.X[i])
            print(pred , self.y[i])

    def accuracy(self):
        correct = 0
        for i in range(len(self.X)):
            conf = self.forward(self.X[i])
            if conf >= 0.5 and self.y[i] == 1:
                correct += 1
            if conf < 0.5 and self.y[i] == 0:
                correct += 1

        return 100 * correct / len(self.X)

    def save_parameters(self, path="model.npz"):
        np.savez(
            path,
            w1=self.w[1],
            w2=self.w[2],
            b1=self.b[1],
            b2=self.b[2]
        )


# derivative bias

model = LogisticRegression()
model.run(3 , 8, 9 , True)
