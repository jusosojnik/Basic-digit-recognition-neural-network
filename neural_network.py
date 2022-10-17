import math
import numpy as np


class NeuralNetwork:
    def __init__(self, num_input, num_hidden, num_output, lr, et):
        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.lr = lr
        self.et = et
        self.b1 = self.b1 = np.random.uniform(-0.5, 0.5, self.num_hidden)
        self.b2 = self.b2 = np.random.uniform(-0.5, 0.5, self.num_output)
        self.w1 = np.random.uniform(-0.5, 0.5, (self.num_input, self.num_hidden))
        self.w2 = self.w2 = np.random.uniform(-0.5, 0.5, (self.num_hidden, self.num_output))
        self.g1 = None
        self.g2 = None

    def activationFunction(self, x):
        return 1 / (1 + math.exp(-x))

    def train(self, data_x, data_y):
        for j in range(1000):
            for i in range(len(data_x)):
                self.predict(data_x[i], True, data_y[i])
        print("Done Training")

    def predict(self, input_x, backprop, input_y=None, ):
        activation = np.vectorize(self.activationFunction)
        o1 = activation(np.matmul(input_x, self.w1) - self.b1)
        o2 = activation(np.matmul(o1, self.w2) - self.b2)
        if backprop:
            self.g2 = o2 * (1 - o2) * (input_y - o2)
            self.g1 = o1 * (1 - o1) * np.matmul(self.w2, self.g2)
            self.backpropagation(input_x, o1)
        else:
            return o2

    def backpropagation(self, input_x, o1):
        nw = np.zeros([len(self.g1), len(input_x)])
        for j, y in enumerate(self.g1):
            for i, x in enumerate(input_x):
                nw[j][i] = self.lr * x * y
        self.w1 = self.w1 + np.transpose(nw)
        nw = np.zeros([len(self.g2), len(o1)])
        for j, y in enumerate(self.g2):
            for i, x in enumerate(o1):
                nw[j][i] = self.lr * x * y
        self.w2 = self.w2 + np.transpose(nw)
        nb = np.zeros(len(self.b1))
        for j, y in enumerate(self.g1):
            nb[j] = self.lr * y * (-1)
        self.b1 = self.b1 + nb
        nb = np.zeros(len(self.b2))
        for j, y in enumerate(self.g2):
            nb[j] = self.lr * y * (-1)
        self.b2 = self.b2 + nb


nn = NeuralNetwork(2, 2, 3, 0.5, 0)
training_x = np.array([[0, 0],
                       [0, 1],
                       [1, 0],
                       [1, 1]])
training_y = np.array([[0, 0, 1],
                       [0, 0, 1],
                       [1, 0, 0],
                       [0, 1, 0]])

