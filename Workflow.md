This project will involve creating a neural network from scratch
for the MNIST data
[X]Understanding of a neural net
[]Create a basic net
    [X] 3 layers - with 1 node per layer
    [X] Define the activation function
        [X] We will use the sigmoid for this case
    [X] Define the cost function
        [X] We will use the MSE for this
    [] Start implementing the basic net
        [X] Initialize random weights between 0 and 1
        [X] Write the feedforward algorithm
            [X] Feed the inputs from layer 1 to the end
            [X] Calculate the MSE
        [X] Write the backpropagation algorithm
            [X] We need to find a recursive format for this algorithm
            [X] Since we want to update many weights in different layers
            [X] We use the batch version - We use the weights from time t-1 
                to update those in time t. 
            [] Fix upward update bug
                [] change self.e to -self.e
            [X] Do the same for bias
[]Expand the net
[]Test the net on real data
[]Complete the project

[]Use the net for research on various other approaches
    [] Batch training on the data