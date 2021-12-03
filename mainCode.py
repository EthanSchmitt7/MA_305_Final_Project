import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import convolution as cv
import time

# Labels
# 0. Earth
# 1. Jupiter
# 2. Mars
# 3. Mercury
# 4. Moon
# 5. Neptune
# 6. Pluto
# 7. Saturn
# 8. Sun
# 9. Uranus
# 10. Venus

# User Inputs
lr = 1e-10  # Learning Rate
numFilters = 10  # How many filters to apply
sizeFilters = 6  # Size of filters
poolSize = 20  # How rows and columns to max pool as a square matrix
hiddenNodes = 100  # Number of hidden nodes
epochs = 100  # Number of epochs

# Bringing in the data
trainData = np.array(pd.read_csv("SimonPlanetsTraining.csv", header=None).T)
testData = np.array(pd.read_csv("SimonPlanetsTesting.csv", header=None).T)

# Randomizing the order
np.random.shuffle(trainData)
np.random.shuffle(testData)

# Extracting input vs. output
rawOutputData = trainData[:, 0].astype(int)  # Need this as one-hot encoding
rawInputData = trainData[:, 1:trainData.shape[1] + 1]  # Need this on a range of 0.01-1, currently 0-255

# Normalizing Inputs
shiftedInputData = rawInputData + abs(rawInputData.min())
rangedInputData = shiftedInputData * 0.99/shiftedInputData.max() + 0.01  # Puts the input data on a new range (0.01-1)

# Defining one hot encoding
diagArray = np.diagflat([np.ones(11)])  # Define the original one hot encoding
oneHotEncoding = (diagArray == 0) * 0.01 + (diagArray == 1) * 0.99  # Make the one hot encoding on new range (0.01-0.99)

# One hot encoding the outputs, and reshapes the inputs as an image
outputData = np.zeros([len(rawOutputData), 11])  # pre-allocating output array
inputData = np.zeros([len(rangedInputData), 1025, int(rangedInputData.shape[1]/1025)])
for trainSet in range(len(rawOutputData)):  # for each training set
    outputData[trainSet, :] = oneHotEncoding[rawOutputData[trainSet]]  # Storing the one hot encoded outputs
    inputData[trainSet] = np.reshape(rangedInputData[trainSet, :], [1025, int(rangedInputData.shape[1]/1025)])  # Reshapes for image processing (1025xminColSize)

# Show image
#plt.imshow(inputData[0], cmap="gray")
#plt.show()

# Generating Filters
filters = np.random.rand(numFilters, sizeFilters, sizeFilters)

# MLP Init
weights1 = np.random.rand(filters.shape[0]*(inputData.shape[1]-filters.shape[1]+1)//poolSize*(inputData.shape[2]-filters.shape[2]+1)//poolSize, hiddenNodes)
weights2 = np.random.rand(hiddenNodes, outputData.shape[1])

# Training
loss = np.zeros([1, epochs])
print('--------------------- Start Training ---------------------')
for epoch in range(epochs):
    t = time.time()

    convolution = cv.convo(inputData, filters)
    [Pooled, Stored] = cv.pooling(convolution, inputData, filters, poolSize)

    # Neural Network
    networkInput = Pooled.reshape([Pooled.shape[0], Pooled.shape[1] * Pooled.shape[2] * Pooled.shape[3]])
    networkInput = networkInput*(networkInput > 0)

    # Feed Forward
    hidden = np.maximum(np.dot(networkInput, weights1), 0)
    output = np.dot(hidden, weights2)
    softmax = np.exp(output - np.amax(output, 1)[:, np.newaxis]) / np.sum(np.exp(output - np.amax(output, 1)[:, np.newaxis]), 1)[:, np.newaxis]

    # Backpropagation
    loss[0, epoch] = ((softmax - outputData) ** 2).sum()
    dL_dP = 2 * (softmax - outputData)

    softmax_reshape = np.reshape(softmax, (1, -1))
    dL_dP_reshape = np.reshape(dL_dP, (1, -1))
    d_softmax = (softmax_reshape * np.identity(softmax_reshape.size) - softmax_reshape.T @ softmax_reshape)
    dP_dZ3 = (dL_dP_reshape @ d_softmax).reshape([softmax.shape[0], softmax.shape[1]])
    dZ3_dW2 = hidden
    dZ3_dX2 = weights2
    dX2_dZ2 = hidden > 0
    dX2_dW1 = networkInput
    dW2 = np.dot(dZ3_dW2.T, dL_dP * dP_dZ3)
    dW1 = np.dot(dX2_dW1.T, np.dot(dL_dP * dP_dZ3, dZ3_dX2.T) * dX2_dZ2)

    # Changes to weights
    weights1 += dW1*lr
    weights2 += dW2*lr

    # Changes to Filters
    dZ2_dX1 = weights1
    dinput = np.dot(np.dot(dL_dP * dP_dZ3, dZ3_dX2.T) * dX2_dZ2, dZ2_dX1.T)
    dinput_reshape = np.reshape(dinput, Pooled.shape)  # Reshape dinput
    d_act = Pooled > 0

    newFilters = cv.rconvo(inputData, filters, dinput_reshape*d_act, Stored)

    filterChange = np.zeros_like(filters)
    for f in range(filters.shape[0]):
        for i in range(inputData.shape[0]):
            filterChange[f] = filterChange[f] + newFilters[i, f]

        filters[f] = filters[f] + filterChange[f]*lr

    # End training step, report stats
    elapsed = time.time() - t
    print('Epoch: ', epoch, '   Loss: ', round(loss[0, epoch], 2), '   Time: ', round(elapsed, 2))

# Plot the trajectory of the loss throughout the training to see how the network learned
plt.plot(np.arange(1, loss.size+1), loss.T)
plt.show()
print()
