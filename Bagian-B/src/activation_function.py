import numpy as np

class ActivationFunction:
    def __init__(self, activation_function):
        if activation_function == 'sigmoid':
            self.name = 'sigmoid'
            self.function = lambda net: self.sigmoid(net)
            self.error_term_output = lambda output, target: self.diffSigmoidActvFunc(output) * (target - output)
            self.error_term_hidden = lambda output, error_term_output, weights: self.diffSigmoidActvFunc(output) * np.dot(error_term_output, weights.T)
        elif activation_function == 'relu':
            self.name = 'relu'
            self.function = lambda net: self.relu(net)
            self.error_term_output = lambda output, target: self.diffReluActvFunc(output) * (target - output)
            self.error_term_hidden = lambda output, error_term_output, weights: self.diffReluActvFunc(output) * np.dot(error_term_output, weights.T)
        elif activation_function == 'linear':
            self.name = 'linear'
            self.function = lambda net: self.linear(net)
            self.error_term_output = lambda output, target: self.diffLinearActvFunc() * (target - output)
            self.error_term_hidden = lambda output, error_term_output, weights: self.diffLinearActvFunc() * np.dot(error_term_output, weights.T)
        elif activation_function == 'softmax':
            self.name = 'softmax'
            self.function = lambda net: self.softmax(net)
            self.error_term_output = lambda output, target: self.calcErrorOutputSoftmax(output, target)
            self.error_term_hidden = lambda output, error_term_output, weights: self.calcErrorHiddenSoftmax(weights, error_term_output, output)

    def sigmoid(self, net):
        return 1 / (1 + np.exp(-net))

    def linear(self, net):
        return net

    def relu(self, net):
        return np.maximum(0, net)

    def softmax(self, net):
        exp_values = np.exp(net - np.max(net, axis=1, keepdims=True))
        return exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def diffLinearActvFunc(self):
        return 1

    def diffSigmoidActvFunc(self, output):
        return output * (1 - output)

    def diffSoftmaxActvFunc(self, p, j, c):
        if j != c:
            return p
        else:
            return -(1 - p)

    def diffReluActvFunc(self, net):
        return np.where(net > 0, 1, 0)

    def diffLossFuncCrossEntropy(self, p, j):
        return -(1 / (p[j] * np.log(10)))

    def calcErrorOutputSoftmax(self, p, target):
        return self.diffSoftmaxActvFunc(p) * self.diffLossFuncCrossEntropy(p, target)

    def calcErrorHiddenSoftmax(self, weight, nextErr, output):
        sigma = np.sum(weight * nextErr)
        return self.diffSoftmaxActvFunc(output) * sigma

    def get_name(self):
        return self.name

    def get_activation_function(self):
        return self.function

    def get_error_term_output(self):
        return self.error_term_output

    def get_error_term_hidden(self):
        return self.error_term_hidden