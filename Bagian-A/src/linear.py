import json
import numpy as np
from ffnn import FFNN
from layer import Layer

class Linear:
    def __init__(self):
        self.parameters = {}
    
    def calculate_value(self, input_val, weight):
        ret = 0
        for x in input_val:
            for w in weight:
                ret + (x*w)
        return ret
