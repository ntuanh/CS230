import numpy as np

class Evaluation:
    def __init__(self):
        pass

    def train_accuracy(self):
        pred = self.forward(self.X)
        pred = (pred > 0.5).astype(int)
        return np.mean(pred == self.y)

    def accuracy(self):
        pred = self.forward(self.X_test)
        pred = (pred > 0.5).astype(int)
        return np.mean(pred == self.y_test)

    def compute_cost(self, Y):
        m = Y.shape[1]
        A = np.clip(self.a[self.L], 1e-8, 1 - 1e-8)

        cross_entropy = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m

        # Add this L2 term to match your backward logic
        l2_term = (self.lambd / (2 * m)) * sum(np.sum(np.square(self.w[i])) for i in range(1, self.L + 1))

        return cross_entropy + l2_term

    def pack_parameters(self, w_dict, b_dict):
        """Reshapes all W and b matrices into a single big vector theta."""
        theta = []
        for i in range(1, self.L + 1):
            theta.append(w_dict[i].flatten())
            theta.append(b_dict[i].flatten())
        return np.concatenate(theta)

    def unpack_parameters(self, theta):
        """Converts big vector theta back into W and b dictionaries."""
        w_new = {}
        b_new = {}
        start = 0
        for i in range(1, self.L + 1):
            # Extract W
            w_shape = self.w[i].shape
            w_size = self.w[i].size
            w_new[i] = theta[start:start + w_size].reshape(w_shape)
            start += w_size

            # Extract b
            b_shape = self.b[i].shape
            b_size = self.b[i].size
            b_new[i] = theta[start:start + b_size].reshape(b_shape)
            start += b_size
        return w_new, b_new

    def checking_gradient(self, X, Y, epsilon=1e-7):
        # 1. Get analytical gradients (d_theta)
        # We run forward and backward once to get the gradients from your backprop
        self.forward(X , check_gradient=True )

        # 2. Pack current parameters into theta
        theta = self.pack_parameters(self.w, self.b)
        num_parameters = theta.shape[0]
        grad_approx = np.zeros(num_parameters)

        print(f"Checking gradients for {num_parameters} parameters...")

        for i in range(num_parameters):
            # J(theta + epsilon)
            theta_plus = np.copy(theta)
            theta_plus[i] += epsilon
            w_p, b_p = self.unpack_parameters(theta_plus)

            # Temporary swap to compute loss
            orig_w, orig_b = self.w, self.b
            self.w, self.b = w_p, b_p
            self.forward(X)
            loss_plus = self.compute_cost(Y)

            # J(theta - epsilon)
            theta_minus = np.copy(theta)
            theta_minus[i] -= epsilon
            w_m, b_m = self.unpack_parameters(theta_minus)
            self.w, self.b = w_m, b_m
            self.forward(X)
            loss_minus = self.compute_cost(Y)

            # Recover original weights
            self.w, self.b = orig_w, orig_b

            # Numerical Gradient
            grad_approx[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # 3. Compare (Assuming you have d_theta from your backward pass)
        # difference = np.linalg.norm(d_theta - grad_approx) / ...

    def get_gradients_vector(self):
        """Packs calculated dw and db into a single vector."""
        out = []
        for i in range(1, self.L + 1):
            out.append(self.grads[f"dw{i}"].flatten())
            out.append(self.grads[f"db{i}"].flatten())
        return np.concatenate(out)

    def run_gradient_check(self, x_sample, y_sample):
        epsilon = 1e-7

        # 1. Get Backprop Gradients (d_theta)
        self.forward(x_sample , check_gradient = True )
        self.backward(y_sample, update=False)  # Don't update weights!
        d_theta = self.get_gradients_vector()

        # 2. Get Numerical Gradients (grad_approx)
        theta = self.pack_parameters(self.w, self.b)
        grad_approx = np.zeros(len(theta))

        print(f"Starting check on {len(theta)} parameters...")

        for i in range(len(theta)):
            # Estimate J(theta + eps)
            theta_plus = np.copy(theta)
            theta_plus[i] += epsilon
            w_p, b_p = self.unpack_parameters(theta_plus)

            old_w, old_b = self.w, self.b  # Swap
            self.w, self.b = w_p, b_p
            self.forward(x_sample , check_gradient = True )
            loss_plus = self.compute_cost(y_sample)

            # Estimate J(theta - eps)
            theta_minus = np.copy(theta)
            theta_minus[i] -= epsilon
            w_m, b_m = self.unpack_parameters(theta_minus)

            self.w, self.b = w_m, b_m
            self.forward(x_sample, check_gradient = True)
            loss_minus = self.compute_cost(y_sample)

            self.w, self.b = old_w, old_b  # Restore

            grad_approx[i] = (loss_plus - loss_minus) / (2 * epsilon)

        # 3. Compare
        numerator = np.linalg.norm(d_theta - grad_approx)
        denominator = np.linalg.norm(d_theta) + np.linalg.norm(grad_approx)
        difference = numerator / denominator

        if difference < 1e-7:
            print(f"Success! Diff: {difference}")
        else:
            print(f"Error in Backprop! Diff: {difference}")