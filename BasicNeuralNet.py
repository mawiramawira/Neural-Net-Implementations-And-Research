import math
import random

class NeuralNet():
    def __init__(self):
        self.no_layers = 2
        #input and target
        self.X = 0
        self.Y = 1
        #inputs
        self.layers = [0,0]
        self.biases = [0,0]
        self.weights = [0,0]
        #w.x + b - help in the backpropagation algorithm
        self.Z = [0,0]
        #error
        self.error = 0 

    #######################################################
    ################# MAIN ALGORITHMS #####################
    #######################################################
    def feedforward(self):
        '''Use the current weights to calculate the error'''
        for i in range(self.no_layers):
            if i == 0:
                z = self.X + self.biases[i]
            else:
                z = (self.layers[i-1] * self.weights[i]) + self.biases[i]
            
            a = self.sigmoid(z)
            self.Z[i] = z
            self.layers[i] = a
        

    def backpropagate(self):
        '''Adjust the weights to reduce the total error. Batch version'''
        lr = 0.1
        new_weights = [0,0]
        new_biases = [0,0]
        a = self.layers[self.no_layers - 1]
        self.error = abs(a - self.Y)
        cost = self.cost(a)
        back_message = -self.error
        
        for i in reversed(range(1,self.no_layers)):
            #how this c changed with the a it was calculated from
            dc_wrt_a = back_message

            #how this a changed with the z it was calculated from 
            da_wrt_z = self.derivative_sigmoid(self.Z[i])

            #how this z changed with the w it was calculated from
            #input from previous layer
            if i == 0:
                dz_wrt_w = self.X
            else:
                dz_wrt_w = self.layers[i-1] 

            #how this z changed with the b it was calculated from
            dz_wrt_b = 1

            #what info will be sent back to previous layers
            back_message = dc_wrt_a*da_wrt_z*self.weights[i]

            dcdw = dc_wrt_a*da_wrt_z*dz_wrt_w
            dcdb = dc_wrt_a*da_wrt_z*dz_wrt_b

            new_weights[i] = self.weights[i] - lr*dcdw
            new_biases[i] = self.biases[i] - lr*dcdb

        #batch version
        self.weights = new_weights
        self.biases = new_biases
        print(a,self.error) 
            
    #######################################################
    ################# HELPER METHODS ######################
    #######################################################
    def sigmoid(self,z):
        '''Sigmoid domain -> [0,1]'''
        return 1/(1+math.exp(-z))

    def derivative_sigmoid(self,z):
        '''derivative of sigmoid'''
        return math.exp(-z)/((1+math.exp(-z))**2)
    
    def cost(self,x):
        '''The Mean Squared Error. 0.5 to ease differentiation'''
        return 0.5 * (x - self.Y)**2

    def initialize(self):
        '''Initialize the neural net using random weights'''
        self.weights = [random.gauss(0,1) for w in self.weights]
        self.biases = [random.gauss(0,1) for w in self.biases]

nn = NeuralNet()
nn.initialize()

for i in range(10):
    nn.feedforward()
    nn.backpropagate()

