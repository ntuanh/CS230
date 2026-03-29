from datasets import  load_dataset
import numpy as np
import cv2

class Preprocessing:
    def __init__(self):
        print("preparing the dataset ")
        self.dataset = load_dataset("Bingsu/Cat_and_Dog")
        self.data_train = self.dataset["train"]
        self.data_test = self.dataset["test"]

        self.X_train = []
        self.X_test = []

        # self.Y_train = self.data_train['labels']
        # self.Y_test = self.data_test['labels']
        self.Y_train = []
        self.Y_test = []

        self.target_size = (16 , 16)


    def load_data(self):
        pass

    def resize_images(self):
        resized_list = []
        for img in self.data_train['image']:
            # 1. Convert PIL image to NumPy array
            img_array = np.array(img)

            # 2. Resize using OpenCV
            # Ensure self.target_size is (width, height)
            resized_img = cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_LINEAR)
            resized_img = resized_img.reshape(-1)

            resized_list.append(resized_img)
        self.X_train = np.array(resized_list).T

        resized_list = []
        for img in self.data_test['image']:
            # 1. Convert PIL image to NumPy array
            img_array = np.array(img)

            # 2. Resize using OpenCV
            # Ensure self.target_size is (width, height)
            resized_img = cv2.resize(img_array, self.target_size, interpolation=cv2.INTER_LINEAR)
            resized_img = resized_img.reshape(-1)

            resized_list.append(resized_img)

        # 3. Final conversion to a single 4D NumPy array (batch, h, w, c)
        self.X_test = np.array(resized_list).T

    def one_hot_output(self):
        self.Y_train = np.array(self.data_train['labels']).reshape(-1 , 1).T
        self.Y_test = np.array(self.data_test['labels']).reshape(-1 , 1).T
        # print(self.Y_train.shape)
        # print(self.Y_test.shape)

    def data(self):
        self.resize_images()
        self.one_hot_output()

        self.X_train = self.normalization(self.X_train).T
        self.X_test = self.normalization(self.X_test)
        self.Y_train = self.Y_train.T

        rng = np.random.default_rng(seed=42)
        indices = rng.permutation(8000)
        self.X_train = self.X_train[indices]
        self.Y_train = self.Y_train[indices]

        self.X_train = self.X_train.T
        self.Y_train = self.Y_train.T

        # print(self.X_train.shape)
        # print(self.X_test.shape)
        # print(self.Y_train[0:20])
        # print(self.Y_test.shape)

        return self.X_train , self.X_test , self.Y_train , self.Y_test

    # scaling data
    def normalization(self , X):
        X = X / 255.0
        mean = np.mean(X)
        std = np.std(X)

        X = ( X - mean )/ std
        return X


    # [ Utils ] start
    def get_input_size(self):
        return self.target_size[0] * self.target_size[1]*3


# preprocessing = Preprocessing()
# x_train , _ , _ , _ = preprocessing.data()
# # x_train = preprocessing.normalization(x_train)
# print(x_train[0][0])

