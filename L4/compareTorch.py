import torch
import torch.nn as nn
import torch.optim as optim
import time
import numpy as np
from DNN import DNN


class TorchDNN(nn.Module):
    def __init__(self, input_dim): # Add input_dim here
        super(TorchDNN, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 8),  # Automatically matches X_train
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.Tanh(),
            nn.Linear(4, 2),
            nn.Sigmoid(),
            nn.Linear(2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)


def run_pytorch_benchmark(X_train, y_train, epochs=100, batch_size=32, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"PyTorch running on: {device}")

    # 1. Prepare data
    # Transpose so shape becomes (Samples, Features)
    X_tensor = torch.FloatTensor(X_train.T).to(device)
    y_tensor = torch.FloatTensor(y_train.T).to(device)

    # 2. Detect dimensions dynamically
    # X_tensor.shape[1] will be 3 based on your error
    input_dim = X_tensor.shape[1]
    print(f"Detected Input Dimensions: {input_dim}")

    # 3. Initialize model with detected dimension
    model = TorchDNN(input_dim=input_dim).to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    start_time = time.time()

    for epoch in range(epochs):
        permutation = torch.randperm(X_tensor.size(0))
        epoch_loss = 0

        for i in range(0, X_tensor.size(0), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x, batch_y = X_tensor[indices], y_tensor[indices]

            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if epoch % 20 == 0:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {epoch_loss / (X_tensor.size(0) / batch_size):.4f}")

    end_time = time.time()
    return end_time - start_time
# 1. Run your NumPy Model
print("--- Starting NumPy Benchmark ---")
model_numpy = DNN(learning_rate=0.001, batch=32)
# Ensure your data is loaded
numpy_time = model_numpy.train(epochs=100) # Ensure your train() returns time

# 2. Run the PyTorch Model
print("\n--- Starting PyTorch Benchmark ---")
pytorch_time = run_pytorch_benchmark(model_numpy.X, model_numpy.y, epochs=100)

print("\n" + "="*30)
print(f"NumPy Training Time:   {numpy_time:.2f}s")
print(f"PyTorch Training Time: {pytorch_time:.2f}s")
print(f"Speedup Factor:        {numpy_time / pytorch_time:.1f}x")
print("="*30)