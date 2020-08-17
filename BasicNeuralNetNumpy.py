import numpy as np
import math

class NeuralNet():
    def __init__(self,shapeLayers,input,target):
        self.A = self.initializeLayers(shapeLayers,input)
        self.noLayers = len(shapeLayers)
        #input and target
        self.Y = np.array(target)
        #inputs
        self.biases = np.zeros(len(shapeLayers) - 1)
        self.weights = self.initializeWeights(shapeLayers)

    #######################################################
    ################# MAIN ALGORITHMS #####################
    #######################################################
    def feedforward(self):
        '''Use the current weights to calculate the error'''
        for i in range(self.noLayers - 1):  
            layer = self.A[i]
            forwardResult = np.matmul(layer,self.weights[i]) + self.biases[i]
            self.A[i+1] = self.sigmoid(forwardResult)

    def backPropagate(self):
        '''Using sigmoid activation'''
        '''Adjust the weights to reduce the total error. Batch version'''
        error = np.array(np.mean(self.Y - self.A[-1]))
        lr = .1
        backMessage = -error
        
        for i in reversed(range(1,self.noLayers)):
            curLayer = self.A[i]
            prevLayer = self.A[i-1]
            #how c changes with the a it was calculated from
            dcwrta = backMessage
            #how a changes with the z it was calculated from 
            dawrtz = self.derivativeSigmoid(self.inverseSigmoid(curLayer)).reshape((len(curLayer),1))
            #how z changes with the w it was calculated from - input from previous layer  
            dzwrtw = prevLayer.reshape((len(prevLayer),1))
    
            dcdw = dcwrta * np.matmul(dzwrtw,dawrtz.T)

            #backMessage = np.mean(dcwrta* np.matmul(dawrtz,self.weights[i-1]))

            self.weights[i-1] = self.weights[i-1] - lr*dcdw

            #print(self.weights)

        print(error,self.A[-1])

    #######################################################
    ################# HELPER METHODS ######################
    #######################################################

    def initializeLayers(self,shapeLayers,input):
        '''Will initialize the layers as per the input and shape of the layers'''
        a = []

        for i in range(len(shapeLayers)):
            arr = np.zeros(shapeLayers[i])

            if i == 0:
                arr = np.array(input)
            
            a.append(arr)
            
        return a

    def initializeWeights(self,shapeLayers):
        '''Will initialize the weights as per the shape of the layers'''
        a = []

        for i in range(1,len(shapeLayers)):
            cur_layer_shape = shapeLayers[i]
            prev_layer_shape = shapeLayers[i-1]
            #mu, sigma, shape
            arr = np.random.normal(0,1,(prev_layer_shape,cur_layer_shape))
            #for testing purposes
            #if i == len(shapeLayers) - 1:
                #arr = np.array([(2,1), (1,1)])
            #if i == 1:
                #arr = np.array([(1,0)])

            a.append(arr)
            
        return a 

    def sigmoid(self,z):
        '''Sigmoid domain -> [0,1]'''
        return 1/(1+np.exp(-z))

    def inverseSigmoid(self,a):
        '''inverse of a to get back z. Used in backprop'''
        return -np.log((1-a)/a)

    def derivativeSigmoid(self,z):
        '''derivative of sigmoid'''
        return np.exp(-z)/((1+np.exp(-z))**2)


nn = NeuralNet(shapeLayers = [1,2,3],input = [0.1], target = [1])

for i in range(10):
    nn.feedforward()
    nn.backPropagate()
