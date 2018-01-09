import numpy as np

def tanh(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1.0 - np.tanh(x)**2

def logistic(x):
    return 1/(1 + np.exp(-x))

def logistic_derivative(x):
    return logistic(x)*(1-logistic(x))

def readTrainingData(filename):
    inputs = []
    outputs = []
    with open(filename) as file:
        for line in file:
            values = [int(value) for value in line.strip().split()]
            inputs.append(np.array(values[:-1]))
            outputs.append(np.array(values[-1]))
    return inputs, outputs

class NeuralNetwork:
    weights = []
    inputs = []
    outputs = []

    def __init__(self, layers):
        self.number_of_layers = len(layers)

        last_layer = 0
        for l in range(self.number_of_layers):
            current_layer = layers[l]
            is_first_layer = (l == 0)

            
            self.inputs.append(np.zeros(current_layer))
            self.outputs.append(np.zeros(current_layer + 1))

            if not is_first_layer:
                self.weights.append(np.ones([current_layer, last_layer + 1]))

            last_layer = current_layer

    def set_input(self, input_values):
        self.outputs[0] = np.append(input_values, 1)

    def get_output(self):
        # returns last layer except bias node
        return self.outputs[-1][:-1]
    
    def propagate(self):
        for l in range(self.number_of_layers - 1):
            self._propagate_layer(l)
            self.outputs[l + 1] = self._activate_neurons(self.inputs[l + 1])

    def _propagate_layer(self, l):
        weights = self.weights[l]
        input_values = self.outputs[l]
        output_values = np.matmul(weights, input_values)
        self.inputs[l + 1] = output_values
    
    def _activate_neurons(self, weighted_sum):
        outputs = np.apply_along_axis(logistic, 0, weighted_sum)
        return np.append(outputs, 1)

    def print_debug(self):
        for l in range(self.number_of_layers):
            print ("------------ Layer #{} ------------".format(l))
            is_first_layer = (l == 0)
                
            if not is_first_layer:
                print("# Weights")
                print(self.weights[l - 1])
            
            print("# Inputs")
            print(self.inputs[l])

            print("# Outputs")
            print(self.outputs[l])


INPUTS, OUTPUTS = readTrainingData('training.txt')

NN = NeuralNetwork([2, 2, 1])
NN.set_input(INPUTS[3])
NN.propagate()
print(NN.get_output())
