import numpy as np
import random

class Build:
    def __init__(self , n):
        self.n = n

    def get_tripple_point(self):
        return random.randint(1 , 10) , random.randint(1 , 10) , random.randint(1 , 10)

    def save_txt(self , data):
        with open("datasets/data.txt", "a") as file:
            file.writelines(data)

    def get_ouput(self):
        a , b, c = self.get_tripple_point()
        edge_max = max(a , b, c)
        edge_min = min(a , b, c)
        edge_med = a + b + c - edge_max - edge_min
        if edge_max > edge_med + edge_min :
            output = 1
        else :
            output = 0
        data = str(a) + " " + str(b) + " " + str(c) + " " + str(output) + "\n"
        self.save_txt(data)

    def run(self):
        for i in range(self.n):
            self.get_ouput()

    def read_data(self):
        path = "datasets/data.txt"
        x = []
        y = []
        with open(path, "r") as file:
            for line in file:
                line = line.strip()
                a , b , c, d = line.split()
                new_data = [int(a), int(b), int(c)]
                for i in range(len(new_data)):
                    new_data[i] *= 0.1
                x.append([new_data])
                y.append(int(d))
        return x , y


build_data = Build(10000)
# build_data.run()
x , y = build_data.read_data()
# print(x)