# Linear Regression using Gradient Descent

import numpy as np

# Test Params
ma = 10.6 # Actual Slope Value
ca = 12.2 # Actual Offset Value
epochs = 1000 # Number of Epochs

# Run Params
m = 0 # Run Slope Value
c = 0 # Run Actual Value
learning_rate = 0.01

# Value Generator
x = np.random.randn(10) # Creating Random x Values
y = ma*x + ca

# Loss Function = Summation(1/N(yhat - (mx + c))**2) 
# To find Minima of this point that we assume is an inverted bell curve, we calculate the gradient which points to the highest increase
# per smallest increment and move opposite to it by a magnitude of average gradient.

# Descent Function
def descend(x,y,m,c,learning_rate):
    dldm = 0.0 # Gradient of the loss function w.r.t Slope
    dldc = 0.0 # Gradient of the loss function w.r.t Offset
    N = x.shape[0] # Total number of elements for averaging
    for xi,yi in zip(x,y):
        dldm += -2*xi*(yi - (m*xi+c)) # Calculation
        dldc += -2*(yi - (m*xi+c)) # Calculation
    m = m - learning_rate*(1/N)*dldm # Updation
    c = c - learning_rate*(1/N)*dldc # Updation
    return m,c

# Model Creation
for epoch in range(epochs):
    m,c = descend(x,y,m,c,learning_rate) # Take a descent step and update
    yhat = m*x + c # Actual value
    loss = np.sum((yhat - y)**2,axis=0)/x.shape[0] # Loss
    print(f"Epoch: {epoch}, Loss: {loss}, Slope: {m}, Offset: {c}")

print(f"\n Final Model is, y = {m} * x + {c}")