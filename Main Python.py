import numpy as np

class NeuralNetwork():

    def __init__(self):
        np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((5, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_input, training_output, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_input)
            error = training_output - output
            adjustments = np.dot(training_input.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))
                              
        return output
                              
if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random synaptic weights")

    print(neural_network.synaptic_weights)

    training_input = np.array([[0.435,7.404,1,2,1],
                                [0.023,0.927,1,1,2],
                                [6.13,14.954,1,1,3],
                                [5.48,6.83,2,1,4]])

    training_output = np.array([[.22,.11,.12,.07]]).T
          

    neural_network.train(training_input, training_output, 100000)

    print("Synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    while True:
        
        A = str(input("Input 1: "))
        B = str(input("Input 2: "))
        C = str(input("Input 3: "))
        D = str(input("Input 4: "))
        E = str(input("Input 5: "))

        print("New situation: input data = ", A, B, C, D, E)
        print("Output data: ")
        print(neural_network.think(np.array([A, B, C, D, E])))

    

          

    

            

        
