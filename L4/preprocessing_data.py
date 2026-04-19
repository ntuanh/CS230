import numpy as np


class TriangleData:
    def __init__(self, num_samples=1000, test_size=0.2, max_length=10.0, normalize=True):
        """
        Args:
            num_samples (int): Total number of data points to generate.
            test_size (float): Proportion of the dataset to include in the test split.
            max_length (float): Maximum possible length for a randomly generated edge.
            normalize (bool): If True, scales the X features to a range of [0, 1].
        """
        # 1. Generate all random edges (initially shape: size, 3)
        X_all = np.random.rand(num_samples, 3) * max_length

        # 2. Calculate labels using the Triangle Inequality Theorem
        a = X_all[:, 0]
        b = X_all[:, 1]
        c = X_all[:, 2]
        is_triangle = (a + b > c) & (a + c > b) & (b + c > a)

        # Reshape y to (size, 1) initially so it stays 2D
        y_all = is_triangle.astype(int).reshape(-1, 1)

        # 3. Normalize the data (Min-Max scaling to [0, 1])
        if normalize:
            X_all = X_all / max_length

        # 4. Shuffle the dataset before splitting
        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X_all = X_all[indices]
        y_all = y_all[indices]

        # 5. Calculate the split index
        split_idx = int(num_samples * (1 - test_size))

        # 6. Transpose (.T) and assign so shapes become (dimension, size)
        self.X = X_all[:split_idx].T  # Shape: (3, training_size)
        self.y = y_all[:split_idx].T  # Shape: (1, training_size)
        self.X_test = X_all[split_idx:].T  # Shape: (3, testing_size)
        self.y_test = y_all[split_idx:].T  # Shape: (1, testing_size)

    def data(self):
        return self.X , self.X_test , self.y , self.y_test


# # --- Example Usage ---
# if __name__ == "__main__":
#     # Create dataset
#     data = TriangleData(num_samples=1000, test_size=0.2)
#
#     # Check the newly transposed shapes
#     print("Training Data Shapes (Dimension, Size):")
#     print(f"X: {data.X.shape}")
#     print(f"y: {data.y.shape}")
#
#     print("\nTesting Data Shapes (Dimension, Size):")
#     print(f"X_test: {data.X_test.shape}")
#     print(f"y_test: {data.y_test.shape}")