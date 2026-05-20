import numpy as np
import torch


class Convolutional:
    def __init__(self , pad = 0 , stride = 1 ):
        self.pad = pad
        self.stride = stride


    def position_compute(self , sub_mtrxA , sub_mtrxB):
        return np.sum(sub_mtrxA * sub_mtrxB)

    def con3d(self , tensorA , filter):         # single matrix A with single matrix B
        filter_size = filter.shape[2]
        tensor_size = tensorA.shape[2]
        output_size = (tensor_size - filter_size + 2*self.pad) // self.stride + 1
        res_mtrx = np.zeros((tensorA.shape[0] , filter.shape[0] , output_size , output_size))

        for bi , batch in enumerate(tensorA):
            for fi , f in enumerate(filter):
                padded_tensor = np.zeros((tensorA.shape[1] , tensor_size + self.pad * 2, tensor_size + self.pad * 2))
                padded_tensor[: , self.pad : self.pad + tensor_size , self.pad : self.pad + tensor_size] = batch
                for i in range(0 , output_size):
                    for j in range(0 , output_size ):

                        res =self.position_compute(padded_tensor[:, i*self.stride:i*self.stride+filter_size ,j*self.stride : j*self.stride + filter_size],f)
                        res_mtrx[bi][fi][i][j] = res
        return res_mtrx






A = np.arange(1, 76).reshape(1 ,3, 5, 5)  # Sequential numbers 1 to 75
B = np.arange(1, 55).reshape(2, 3, 3, 3)          # All 2s
#
# print(A)
# print(B)

conv = Convolutional(pad=1 , stride=3)


print("result of tensor conv")
print(conv.matrix_compute(A,B))


import torch
import torch.nn.functional as F

# 1. Recreate your exact input and kernel tensors
x = torch.arange(1, 76, dtype=torch.float32).reshape(1, 3, 5, 5)
w = torch.arange(1, 55, dtype=torch.float32).reshape(2, 3, 3, 3)

# 2. Compute using correct stride and padding
result = F.conv2d(x, w, padding=1, stride=3)
print(result.squeeze())




