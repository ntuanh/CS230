import os , random , sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from L4.Activation_funcs import ActivationFunctions
from L4.Evaluation import Evaluation
from L4.Initialization import Initialization
from L4.Optimization import Optimization
from L4.Regularization import Regularization


# build QDN from scratch

class DQN(ActivationFunctions , Initialization , Evaluation , Regularization , Optimization):
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

        # setup for Deep Q-Learning
        self.reward = [0, 0, 0, 0, 0 , 0 , 10]   # start from 1 to 4
        self.total_state = 6
        self.gamma = 0.9
        self.Q_epsilon = 0.2

        # replay memory [] # [curr state , action (0 for left and 1 for right  , reward , next state , done ] ex: [2 , 1 , 0 , 3 , 0]
        self.D = []

        # neural network
        self.layers = [self.total_state, 6, 2]
        self.act_funcs = ["None", "relu", "relu"]
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
        self.p = 0.0    # fraction of dropout
        # print(f"Dropout ratio {self.p}")
        self.dropped_nodes = {}     # storage lists index dropped node

        # display
        self.mean_acc = 0

        self.grads = {}

    def ont_hot_vector(self , state):
        vector = [0 for _ in range(self.total_state)]
        vector[state-1] = 1
        vector = np.array(vector)
        vector = vector.reshape(self.total_state , 1)
        return vector

    def forward(self, x , check_gradient = False ):
        self.a[0] = x  # (feature, batch)

        for i in range(1, self.L + 1):
            self.z[i] = self.w[i].T @ self.a[i-1] + self.b[i]
            self.a[i] = self.activation_func(self.z[i], i)
            if check_gradient is False:
                rand_matrix = np.random.rand(*self.a[i].shape)
                mask = (rand_matrix < (1 - self.p))
                self.a[i] = (self.a[i] * mask) / (1 - self.p)
        return self.a[self.L]

    def backward(self, x, y, update = True):
        self.forward(x)
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

    def update_weight(self):
        y_batch = None
        s_batch = None
        cnt = 0
        while cnt < self.batch:
            cnt += 1
            s , is_right , r , s_next , done = random.choice(self.D)

            q_next = self.forward(self.ont_hot_vector(s_next))
            if done:
                target_value = r
            else :
                target_value = r + self.gamma * max(q_next[0][0] , q_next[1][0])

            target_y = self.forward(self.ont_hot_vector(s_next)).copy()

            if is_right:
                target_y[1][0] = target_value
            else :
                target_y[0][0] = target_value

            if s_batch is None :
                s_batch = self.ont_hot_vector(s)
            else :
                s_batch = np.concatenate((s_batch , self.ont_hot_vector(s)) , axis=1)

            if y_batch is None :
                y_batch = target_y
            else :
                y_batch = np.concatenate((y_batch , target_y) , axis=1)

        # print(f"target y shape : {y_batch.shape}")
        # print(f"s batch shape : {s_batch.shape}")
        self.backward(s_batch , y_batch)

    def train(self):
        s = 1
        cnt = 0
        cnt_prev = 0

        while True :
            cnt += 1
            pred = self.forward(self.ont_hot_vector(s))
            reward_right = pred[1][0]
            reward_left  = pred[0][0]

            if random.uniform(0 , 1 ) < self.Q_epsilon or reward_right == reward_left:
                action_is_right = random.choice([True, False])
            else :
                action_is_right = reward_right > reward_left

            if action_is_right :
                # 1. setup s next
                s_next = s + 1

                # push to D memory
                reward_right_true = self.reward[s_next]
                temp_d  = [s , action_is_right , reward_right_true , s_next , s_next== self.total_state ]
                self.D.append(temp_d)
            else :
                # 1. setup s next
                s_next = s - 1 if s > 1 else 1

                # push to D memory
                reward_left_true = self.reward[s_next]
                temp_d = [s, action_is_right, reward_left_true, s_next, s_next == self.total_state]
                self.D.append(temp_d)
            # print(f"state : {s}\t action {action_str}\tto state {s_next}")
            # print(f"state {s} || [ {reward_left:.3f} , {reward_right:.3f} ]\taction {action_str}")
            # print(f"reward : {temp_reward}")

            if s_next == self.total_state :
                print(f"\nReach to Treasure ! \t \t count to treasure {cnt - cnt_prev}.")
                for i in range(1 , self.total_state):
                    r = self.forward(self.ont_hot_vector(i))
                    print(f"[{r[0][0]:.3f} , {r[1][0]:.3f}]" , end="\t")
                cnt_prev = cnt
                s = 1
            else :
                s = s_next

            self.update_weight()

            if cnt > 20 :
                self.Q_epsilon = 0.1
            if cnt > 50 :
                self.Q_epsilon = 0
            if cnt > 200 :
                break


model = DQN(learning_rate=0.1, batch=4, verbose=False , my_data=False)
model.init_weights()
model.train()
