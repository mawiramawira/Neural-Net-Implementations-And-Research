This project will involve creating a neural network from scratch for the MNIST data

NOTES:
- each index a, is defined by the same index b and w so a[2] = f(z[2]+w[2]*input from previous layer)

[X]Understanding of a neural net
[X]Create a basic net
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
            [X] Fix upward update bug
                [X] change self.e to -self.e
            [X] Do the same for bias
        [X]Expand the net to several layers with several inputs
            [X]Change weight and bias formatting to prev so weights will be from not to 
                [X]Therefore z[0] = w[-1]x[-1] + b[-1]
    [X]Convert to numpy
        weight matrix rows = previous layer
        weight matrix columns = current layer we are feeding forward to
        so each row shows how the previous layer is connected to each of the nodes in the current layer
    [X]Generalize the net to allow initialization via constructor
    []Understand the cde you've written - especially numpy functions
        []Write documentation for your net

[]Test the net on real data
    []Find a way to collect data on MNIST 
    []Make it manageable with the net you wrote
    []Train and test
    
[]Complete the project

[]Use the net for research on various other approaches
    [] Batch training on the data