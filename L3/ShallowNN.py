import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, Y = make_moons(n_samples=1000, noise=0.2)

class ShallowNN:
    def __init__(self , learning_rate , batch ):
        """ constant parameters """
        self.learning_rate = learning_rate
        self.batch = batch

        # data
        self.X , self.X_test , self.y ,self.y_test = train_test_split(
            X , Y ,
            test_size = 0.2 ,
            random_state = 42
        )


        # weights and bias
        self.n_node_hidden_1 = 4
        self.n_node_input = 2

        self.w = {
            1 : np.random.rand(self.n_node_input , self.n_node_hidden_1),
            2 : np.random.rand(self.n_node_hidden_1 , 1)
        }

        self.b = {
            1 : np.random.rand(self.n_node_hidden_1 , 1),
            2 : np.zeros((1 , 1))
        }

        # hidden layers
        self.n_hidden_layers = 1
        self.z = {
            1 : np.random.rand(self.n_node_hidden_1 , 1) ,
            2 : np.zeros((1 , 1))
        }



    """ Util functions """
    def compute_z(self , w , x , b):
        return w.T @ x + b

    def sigmoid(self , z ):
        return 1 / ( 1 + np.exp(-z))

    def derivative_sigmoid(self , z):
        return np.diag(self.sigmoid(z) * (1 - self.sigmoid(z)))

    """ Main functions """
    def initialization_w(self):
        pass

    def derivative_w2(self , x , y_truth):
        a2 = self.sigmoid(self.z[2])
        a1 = self.sigmoid(self.z[1])

        dz2 = a2 - y_truth

        dw2 = a1 @ dz2.T


        return dw2

    def derivative_w1(self , x, y_truth):
        x = x.reshape(-1 , 1)
        a2 = self.sigmoid(self.z[2])
        a1 = self.sigmoid(self.z[1])

        dz2 = a2 - y_truth
        dz1 = (self.w[2] @ dz2) * (a1 * (1 - a1))

        dw1 = x @ dz1.T

        return dw1

    def derivative_b1(self  , x, y_truth):
        a2 = self.sigmoid(self.z[2])
        a1 = self.sigmoid(self.z[1])

        dz2 = a2 - y_truth
        db = (self.w[2] @ dz2) * (a1 * (1 - a1))   # db = dz1

        return db

    def get_cost(self):
        pass

    def get_accuracy(self):
        cnt_correct_cases = 0
        for i in range(len(self.X_test)):
            predict = self.forward(self.X_test[i])
            if predict >= 0.5 and self.y_test[i] == 1 :
                cnt_correct_cases += 1
            elif predict < 0.5 and self.y_test[i] == 0 :
                cnt_correct_cases += 1

        return cnt_correct_cases / len(self.X_test)

    """ Implement functions """
    def forward(self , x ):
        a = np.array(x)
        a = a.reshape(-1 , 1)
        for layer in self.w:
            zi = self.w[layer].T @ a + self.b[layer]
            temp = self.w[layer].T @ a
            self.z[layer] = zi
            a = self.sigmoid(zi)
        return a

    def backward(self , x , y_truth):
        self.w[2] -= self.learning_rate * self.derivative_w2(x , y_truth)
        self.w[1] -= self.learning_rate * self.derivative_w1(x, y_truth)
        self.b[1] -= self.learning_rate * self.derivative_b1(x , y_truth)

    def run(self):
        for i in range(len(self.X)):
            self.forward(self.X[i])
            self.backward(self.X[i] , self.y[i])

    def test(self):
        pass


""" constant parameters """
learning_rate = 0.01
batch_size = 5

model = ShallowNN(learning_rate , batch_size)
model.run()
print(model.get_accuracy())
# print(Y[2])

# z = np.array([1 , 2])
# z = np.random.rand(2 , 1)
# print(z.shape)
# z = model.sigmoid(z)
# print(z.shape)