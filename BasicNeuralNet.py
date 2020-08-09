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
        self.weights = [[[0,0],[0,0]],[[0,0],[0,0]]]
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
                    z = self.previousLayerToCurrentNode(self.X,j,self.weights[i]) + self.biases[i]
                else:
                    z = self.previousLayerToCurrentNode(self.layers[i-1],j,self.weights[i]) + self.biases[i]

                a = self.sigmoid(z)
                self.Z[i][j] = z
                self.layers[i][j] = a

    def curNodeToPrevLayer(self,cur_node,prev_layer,con_weights):
        '''Backpropagate error from current node to previous layer'''
        print(cur_node,prev_layer,con_weights)

        for i in range(len(prev_layer)):
            #how this z changed with the w it was calculated from
            #input from previous layer
            back_error = prev_layer[i]

        raise SystemExit(0)

    def backpropagate(self):
        '''Adjust the weights to reduce the total error. Batch version'''
        lr = 0.1
        new_weights = [[[0,0],[0,0]],[[0,0],[0,0]]]
        new_biases = [0,0]
        a = self.layers[self.no_layers - 1]
        self.error = mean([abs(x - self.Y[0]) for x in a])
        #cost = self.cost(a)
        back_message = -self.error

        #for each layer
        for i in reversed(range(self.no_layers)):
            cur_layer = self.layers[i]
            back_message_hold = []

            #for each node in current layer
            for j in range(len(cur_layer)):
                #how this c changed with the a it was calculated from
                dc_wrt_a = back_message

                #how this a changed with the z it was calculated from 
                da_wrt_z = self.derivative_sigmoid(self.Z[i][j])

                #how this z changed with the w it was calculated from
                #input from previous layer
                if i == 0:
                    dz_wrt_w = self.X
                else:
                    dz_wrt_w = self.layers[i-1] 

                #how this z changed with the b it was calculated from
                dz_wrt_b = 1

                #for each dz_wrt_w_x in dz_wrt_w that changed the current z
                #since dz_wrt_w = sum(dz_wrt_w_x)
                for k in range(len(dz_wrt_w)):
                    #what info will be sent back to previous layers
                    back_message_hold.append(dc_wrt_a*da_wrt_z*self.weights[i][j][k])

                    dcdw = dc_wrt_a*da_wrt_z*dz_wrt_w[k]
                    dcdb = dc_wrt_a*da_wrt_z*dz_wrt_b

                    new_weights[i][j][k] = self.weights[i][j][k] - lr*dcdw
                    new_biases[i] = self.biases[i] - lr*dcdb
            
            back_message = mean(back_message_hold)

        #batch version
        self.weights = new_weights
        self.biases = new_biases
        print(a,self.error)
            
    #######################################################
    ################# HELPER METHODS ######################
    #######################################################
    def previousLayerToCurrentNode(self,prev_layer,cur_node_pos,con_weights):
        '''
            Calculate the input to current node from previous layer.
            cur_node_pos is position of node in current layer and used to map to weights
        '''
        input = 0

        for i in range(len(prev_layer)):
            input += prev_layer[i] * con_weights[i][cur_node_pos]
        
        return input

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
        self.weights = [[[0.1,0.1],[0.1,0.1]],[[0.1,0.3],[0.7,0.2]]]
        self.biases = [0,0]

nn = NeuralNet()
nn.initialize()

for i in range(100):
    nn.feedforward()
    nn.backpropagate()

