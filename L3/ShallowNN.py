import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

X, Y = make_moons(n_samples=30000, noise=0.2)

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

        # auto
        self.model = {
            "hidden_layers" : [2 , 3 , 1],    # input , nodes , nodes , output
            "activation_functions" : ["sigmoid" , "sigmoid" , "sigmoid" , "sigmoid" ] #, "sigmoid"]
        }

        # weights and bias
        self.w = {}
        self.b = {}
        self.z = {}
        self.a = {}     # store a after forward
        self.d = {}     # derivative

        self.dw = {}    # cumulative of dw
        self.db = {}    # cumulative of db

        self.depth = 0
        self.prev_cost = 0
        self.delta_cost = 0


    """ Util functions """
    def compute_z(self , w , x , b):
        return w.T @ x + b

    # [ Activation functions ] start

    def activation(self , z , hidden_layer_idx):
        hidden_layer = self.model["activation_functions"][hidden_layer_idx - 1]
        if hidden_layer == "sigmoid":
            return self.sigmoid(z)
        else :
            print("not sigmoid")

    def derivative_activation(self , z, hidden_layer_idx):
        hidden_layer = self.model["activation_functions"][hidden_layer_idx - 1]
        if hidden_layer == "sigmoid":
            return self.derivative_sigmoid(z)

    def sigmoid(self , z ):
        return 1 / ( 1 + np.exp(-z))

    def derivative_sigmoid(self, z):
        s = self.sigmoid(z)
        return np.diag((s * (1 - s)).flatten())

    # [ Activation functions ] end

    # [ Main functions ] start
    def initialization_w(self):
        self.depth = len(self.model["hidden_layers"])


        # init weights , bias , z
        for i in range(1 , self.depth):
            self.w[i] = np.random.rand(self.model["hidden_layers"][i-1], self.model["hidden_layers"][i])
            self.b[i] = np.random.rand(self.model["hidden_layers"][i] , 1)
            self.z[i] = np.zeros((self.model["hidden_layers"][i], 1))
            self.dw[i] = 0
            self.db[i] = 0

    def get_cost(self):
        loss = 0
        for i in range(len(self.X)):
            predict = self.forward(self.X[i])
            loss_i = self.y[i] * np.log(predict) + (1 - self.y[i]) * np.log(1 - predict)
            loss -= loss_i

        return loss / len(self.X)

    def get_accuracy(self):
        cnt_correct_cases = 0
        for i in range(len(self.X_test)):
            predict = self.forward(self.X_test[i])
            if predict >= 0.5 and self.y_test[i] == 1 :
                cnt_correct_cases += 1
            elif predict < 0.5 and self.y_test[i] == 0 :
                cnt_correct_cases += 1

        return cnt_correct_cases / len(self.X_test)

    # [ Main functions ] end

    """ Implement functions """
    def forward(self , x ):
        a = np.array(x).reshape(-1 , 1)
        for layer_idx in self.w:
            zi = self.w[layer_idx].T @ a + self.b[layer_idx]
            self.z[layer_idx] = zi
            a = self.sigmoid(zi)
            a = self.activation(zi , int(layer_idx))
            self.a[layer_idx] = a
        return a

    def backward(self , x , y_truth , index):
        self.a[0] = np.array(x).reshape(-1, 1)

        end_idx = self.depth-1
        self.d[f"l_z{end_idx}"] = (self.a[end_idx] - y_truth)
        self.d[f"l_w{end_idx}"] = self.a[end_idx - 1] @ self.d[f"l_z{end_idx}"]

        self.d[f"z{end_idx}_a{end_idx - 1}"] = np.transpose(self.w[end_idx])
        self.d[f"l_a{end_idx - 1}"] = self.d[f"l_z{end_idx}"] @ self.d[f"z{end_idx}_a{end_idx - 1}"]
        # print(self.d[f"l_a{end_idx - 1}"].shape)

        for i in range(end_idx - 1, 0, -1):
            self.d[f"a{i}_z{i}"] = np.array(self.derivative_sigmoid(self.z[i]))  # 4x4
            self.d[f"l_z{i}"] = np.array(self.d[f"l_a{i}"] @ self.d[f"a{i}_z{i}"])  # 1x4
            print(self.a[i - 1].shape)
            self.d[f"l_w{i}"] = np.array(self.a[i - 1] @ self.d[f"l_z{i}"])  # 3 x 4
            # compute dl / dx
            if i > 1:
                self.d[f"z{i}_a{i - 1}"] = np.transpose(self.w[i])
                self.d[f"l_a{i - 1}"] = np.array(self.d[f"l_z{i}"] @ self.d[f"z{i}_a{i - 1}"])

        for i in range(1 , self.depth , 1):
            self.dw[i] += self.d[f"l_w{i}"]
            self.db[i] += np.transpose(self.d[f"l_z{i}"])


        if (index + 1) % self.batch == 0:
            for i in range(1 , self.depth ,1):
                self.w[i] -= self.learning_rate * self.dw[i] / self.batch
                self.dw[i] = 0

                self.b[i] -= self.learning_rate * self.db[i] / self.batch
                self.db[i] = 0



    def run(self):
        self.initialization_w()
        epoch = 1
        threshold = 10
        print(len(self.X))
        for epoch_idx in range(epoch):
            print(f"============ {epoch_idx}")
            # for i in range(len(self.X)):
            for i in range(1):
                self.forward(self.X[i])
                print("self a")
                for idx in self.a:
                    print(self.a[idx].shape)
                print("self a")
                self.backward(self.X[i] , self.y[i] , i)
            cost = self.get_cost()

            if self.prev_cost != 0 :
                self.delta_cost = (1 - self.get_accuracy()) * 100
                if self.delta_cost < threshold :
                    self.learning_rate = self.learning_rate / 10

            print(f'cost {cost} and acc : {self.get_accuracy()}')
            # print(self.w[2])

    def test(self):
        self.a[0] = np.array(self.X[0]).reshape(-1 , 1)
        y_truth = self.y[0]
        self.initialization_w()
        self.forward(self.a[0])
        print(y_truth)


        self.d["l_z3"] = (self.a[3] - y_truth)  # 1x1
        self.d["l_w3"] = self.a[2]  @  self.d["l_z3"]    # loss / w3

        self.d["z3_a2"] = np.transpose(self.w[3])   # 1x4
        self.d["l_a2"] = self.d["l_z3"] @ self.d["z3_a2"]   # 1x1 x 1x4 = 1x4
        print(f"dl / da2 {self.d["l_a2"]}")

        for i in range(2 , 0 , -1):
            self.d[f"a{i}_z{i}"] = np.array(self.derivative_sigmoid(self.z[i]))  # 4x4
            self.d[f"l_z{i}"] = np.array(self.d[f"l_a{i}"] @ self.d[f"a{i}_z{i}"])     # 1x4
            print(f"debug at i  {i}")
            print(np.array(self.a[i-1]).shape)
            print(np.array(self.d[f"l_z{i}"]).shape)
            self.d[f"l_w{i}"] = np.array(self.a[i-1] @ self.d[f"l_z{i}"])           # 3 x 4
            # compute dl / dx
            if i > 1 :
                self.d[f"z{i}_a{i-1}"] = np.transpose(self.w[i])
                self.d[f"l_a{i-1}"] = np.array(self.d[f"l_z{i}"] @ self.d[f"z{i}_a{i-1}"])

        print(f"loss / w3 \n{self.d["l_w3"]}")



""" constant parameters """
learning_rate = 0.1
batch_size = 10

model = ShallowNN(learning_rate , batch_size)
model.run()
# print(model.get_accuracy())
# model.test()
