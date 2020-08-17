import math
import random
from statistics import mean 

class NeuralNet():
    def __init__(self):
        self.no_layers = 2
        #input and target
        self.X = [0.1]
        self.Y = [1]
        #inputs
        self.layers = [[1,1],[0,0]]
        self.biases = [0,0]
        self.weights = [[[0],[0]],[[0,0],[0,0]]]
        #w.x + b - help in the backpropagation algorithm
        self.Z = [[0,0],[0,0]]
        #error
        self.error = 0 

    #######################################################
    ################# MAIN ALGORITHMS #####################
    #######################################################
    def feedforward(self):
        '''Use the current weights to calculate the error'''
        for i in range(self.no_layers):
            cur_layer = self.layers[i]

            for j in range(len(cur_layer)):
                if i == 0:
                    z = self.previousLayerToCurrentNode(self.X,self.weights[i][j]) + self.biases[i]
                else:
                    z = self.previousLayerToCurrentNode(self.layers[i-1],self.weights[i][j]) + self.biases[i]

                a = self.sigmoid(z)
                self.Z[i][j] = z
                self.layers[i][j] = a

    def backpropagate(self):
        '''Adjust the weights to reduce the total error. Batch version'''
        lr = 0.1
        new_weights = [[[0.0],[0.0]],[[0,0],[0,0]]]
        new_biases = [0,0]
        a = self.layers[self.no_layers - 1]
        self.error = mean([abs(x - self.Y[0]) for x in a])
        #cost = self.cost(a)
        back_message = -self.error

        #for each layer
        for i in reversed(range(self.no_layers)):
            cur_layer = self.layers[i]

            #for each node in current layer
            for j in range(len(cur_layer)):
                #how this c changed with the a it was calculated from
                dc_wrt_a = back_message

                #how this a changed with the z it was calculated from 
                da_wrt_z = self.derivative_sigmoid(self.Z[i][j])

                #how this z changed with the W's it was calculated from
                #input from previous layer
                if i == 0:
                    dz_wrt_W = self.X
                else:
                    dz_wrt_W = self.layers[i-1] 

                #how this z changed with the b it was calculated from
                dz_wrt_b = 1
            
                #adjust individual weights
                new_weights_hold, back_message_list = self.adjustWeights(lr,dc_wrt_a,da_wrt_z,dz_wrt_W,self.weights[i][j])

                new_weights[i][j] = new_weights_hold

                #print(self.weights)
                #print(new_weights)

                dcdb = dc_wrt_a*da_wrt_z*dz_wrt_b
                new_biases[i] = self.biases[i] - lr*dcdb
                
                back_message = mean(back_message_list)

        #batch version
        self.weights = new_weights
        self.biases = new_biases
        print(a,self.error)
            
    #######################################################
    ################# HELPER METHODS ######################
    #######################################################
    def previousLayerToCurrentNode(self,prev_layer,con_weights):
        '''
            Calculate the input to current node from previous layer.
            cur_node_pos is position of node in current layer and used to map to weights
        '''
        input = 0

        for i in range(len(prev_layer)):
            input += prev_layer[i] * con_weights[i]

        return input

    def adjustWeights(self,lr,dc_wrt_a,da_wrt_z,dz_wrt_W,con_weights):
        '''Adjust individual weights based on the dz_wrt_W matrix'''
        #since dz_wrt_w = sum(dz_wrt_w_x)
        new_weights = []
        back_message_hold = []

        for i in range(len(dz_wrt_W)):
            #what info will be sent back to previous layers
            back_message_hold.append(dc_wrt_a*da_wrt_z*con_weights[i])
            dcdw = dc_wrt_a*da_wrt_z*dz_wrt_W[i]
            updated_weight = con_weights[i] - lr*dcdw

            new_weights.append(updated_weight)
            
        return new_weights,back_message_hold

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
        self.weights = [[[random.gauss(0,1)],[random.gauss(0,1)]],[[random.gauss(0,1),random.gauss(0,1)],[random.gauss(0,1),random.gauss(0,1)]]]
        self.biases = [random.gauss(0,1),random.gauss(0,1)]

nn = NeuralNet()
nn.initialize()

for i in range(10000):
    nn.feedforward()
    nn.backpropagate()

