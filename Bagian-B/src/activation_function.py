import numpy as np

class ActivationFunction:
    def __init__(self, activation_function):
        if activation_function == 'sigmoid':
            self.name = 'sigmoid'
            self.function = lambda net: 1 / (1 + np.exp(-net))
            self.error_term_output = lambda output, target : output * (1 - output) * (target - output)
            self.error_term_hidden = lambda output, error_term_output, weights : output * (1 - output) * np.dot(error_term_output, weights.T)
            self.loss = lambda output, target : np.sum((target - output) ** 2) / 2
        elif activation_function == 'relu':
            self.name = 'relu'
            self.function = lambda net: np.maximum(0, net)
            self.error_term_output = lambda output, target : np.where(output > 0, 1, 0) * (target - output)
            self.error_term_hidden = lambda output, error_term_output, weights : np.where(output > 0, 1, 0) * np.dot(error_term_output, weights.T)
            self.loss = lambda output, target : np.sum((target - output) ** 2) / 2
        elif activation_function == 'linear':
            self.name = 'linear'
            self.function = lambda net: net
            self.error_term_output = lambda output, target : target - output
            self.error_term_hidden = lambda _, error_term_output, weights : np.dot(error_term_output, weights.T)
            self.loss = lambda output, target : np.sum((target - output) ** 2) / 2
        elif activation_function == 'softmax':
            self.name = 'softmax'
            self.function = lambda net: np.exp(net) / np.sum(np.exp(net))
            self.error_term_output = lambda output, target : self.softmax_error_term(output, target)
            self.error_term_hidden = lambda output, target, _ : self.softmax_error_term(output, target)
            self.loss = lambda output, target : self.softmax_loss(output, target)

    def softmax_error_term(self, output, target):
        correct_label = np.argmax(target, axis=1)
        error_term = output.copy()
        for i in range(len(correct_label)):
            error_term[i][correct_label[0]] = - (1 - output[i][correct_label[0]])
        return error_term

    def softmax_loss(self, output, target):
        target_label = np.argmax(target, axis=1)
        return -np.sum(np.log(output[np.arange(len(target_label)), target_label]))

    def get_name(self):
        return self.name
    
    def get_activation_function(self):
        return self.function
    
    def get_error_term_output(self):
        return self.error_term_output
    
    def get_error_term_hidden(self):
        return self.error_term_hidden
    
    def get_loss(self):
        return self.loss