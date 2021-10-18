# NN BackPropogation using Excel

Backpropagation is used for training a neural network. Backpropagation adjusts the weights so that the neural network can map inputs to outputs. This backpropagation exercise shows the example calculations using excel sheet to help understand backpropagation. A neural network with two inputs, two hidden neurons, two output neurons are used and biases are ignored.

Initial weights,

    w1 = 0.15	w2 = 0.2	w3 = 0.25	w4 = 0.3
    w5 = 0.4	w6 = 0.45	w7 = 0.5	w8 = 0.55
# Learning rate graph
![BP_graph](https://user-images.githubusercontent.com/90888045/137781125-52ec7a1d-af91-4d49-ae04-c09540d3852d.png)

# ScreenShot
![Screenshot (114)](https://user-images.githubusercontent.com/90888045/137781193-56e23f76-d0fe-497c-8bbc-f899b30dcb99.png)

Given two inputs with values 0.05 and 0.10 and two exprected outputs 0.01 and 0.99.

## Forward Propogation

Pass the above inputs through the network by multiplying the inputs to the weights and calculate the h1 and h2
    
      h1 =w1*i1+w2+i2
      h2 =w3*i1+w4*i2
      
The output from the hidden layer neurons (h1 and h2) are passed to activated neurons using sigmoid activation function. Activation functions add non-linearity to the neural network.

      a_h1 = σ(h1) = 1/(1+exp(-h1))
      a_h2 = σ(h2) = 1/(1+exp(-h2))

The process is repeated for the output layer neurons, using the output from the hidden layer actiavted neurons as inputs.

      o1 = w5 * a_h1 + w6 * a_h2
      o2 = w7 * a_h1 + w8 * a_h2
      
      a_o1 = σ(o1) = 1/(1+exp(-o1))
      a_o2 = σ(o2) = 1/(1+exp(-o2))
      
Calculate the error for each output neurons (a_o1 and a_o2) using the squared error function and sum them up to get the total error (E_total)

## Calculating the Error (Loss)
      
    E1 = ½ * ( t1 - a_o1)²
    E2 = ½ * ( t2 - a_o2)²
    E_Total = E1 + E2

Note:  1/2 is included so that exponent is cancelled when error term is differenciated.

    
## Back Propogation

Back propogation is when the network learns and improves by updating the weights with the goal of minimizing error

Calculate the partial derivative of E_total with respect to w5 

    δE_total/δw5 = δ(E1 +E2)/δw5
    
    δE_total/δw5 = δ(E1)/δw5       # removing E2 as there is no impact from E2 wrt w5	
                 = (δE1/δa_o1) * (δa_o1/δo1) * (δo1/δw5)	# Using Chain Rule
                 = (δ(½ * ( t1 - a_o1)²) /δa_o1= (t1 - a_o1) * (-1) = (a_o1 - t1))
                    * (δ(σ(o1))/δo1 = σ(o1) * (1-σ(o1)) = a_o1                   
                    * (1 - a_o1 )) * a_h1                                       
                 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1


Calculate the partial derivative of E_total with respect to w6, w7, w8.

    δE_total/δw5 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h1
    δE_total/δw6 = (a_o1 - t1 ) *a_o1 * (1 - a_o1 ) * a_h2
    δE_total/δw7 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h1
    δE_total/δw8 = (a_o2 - t2 ) *a_o2 * (1 - a_o2 ) * a_h2


Do back propogation through the hidden layers

    δE_total/δa_h1 = δ(E1+E2)/δa_h1 
                   = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7
                   
    δE_total/δa_h2 = δ(E1+E2)/δa_h2 
                   = (a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8
                   
Calculate the partial derivative of E_total with respect to w1, w2, w3 and w4 using chain rule   

    δE_total/δw1 = δE_total/δw1 = δ(E_total)/δa_o1 * δa_o1/δo1 * δo1/δa_h1 * δa_h1/δh1 * δh1/δw1
                 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i1
                 
    
    δE_total/δw2 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w5 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w7) * a_h1 * (1- a_h1) * i2
    δE_total/δw3 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i1
    δE_total/δw4 = ((a_o1 - t1) * a_o1 * (1 - a_o1 ) * w6 + (a_o2 - t2) * a_o2 * (1 - a_o2 ) * w8) * a_h2 * (1- a_h2) * i2


Update the weights by subtracting gradient times learning rate from current weight 

        w1 = w1 - learning_rate * δE_total/δw1
        w2 = w2 - learning_rate * δE_total/δw2
        w3 = w3 - learning_rate * δE_total/δw3
        w4 = w4 - learning_rate * δE_total/δw4
        w5 = w5 - learning_rate * δE_total/δw5
        w8 = w6 - learning_rate * δE_total/δw6
        w7 = w7 - learning_rate * δE_total/δw7
        w8 = w8 - learning_rate * δE_total/δw8


## Error Graph for different Learning rates

Link to Excel Sheet - https://github.com/satyaNekkantiCompVison/ExtensiveVisionAI/tree/main/S4_BackPropagation/BackPropagation.xlsx

Below is the error graph when we change the learning rates 0.1, 0.2, 0.5, 0.8, 1.0, 2.0


For small learning rate the loss drops very slowly and takes longer to converge.
Learning rate should be optimal (neither too low nor too high) for learning to occur.

# EVA7
