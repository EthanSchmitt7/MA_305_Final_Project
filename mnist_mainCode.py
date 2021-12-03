import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import convolution as cv
import time

tTill = 10  # How many of inputs to run through
lr = 1e-7

# Bringing in the data and formatting
df = pd.read_csv('mnist_train.csv')  # Import data into pandas DF
trainData = np.array(df)  # Convert pandas DF into a regular array

rawOutputData = trainData[0:tTill, 0]  # Need this as one-hot encoding
rawInputData = trainData[0:tTill, 1:trainData.shape[0] + 1]  # Need this on a range of 0.01-1, currently 0-255
rangedInputData = rawInputData * 0.99 / 255 + 0.01  # Puts the input data on a new range (0.01-1)

diagArray = np.diagflat([np.ones(10)])  # Define the original one hot encoding
oneHotEncoding = (diagArray == 0) * 0.01 + (diagArray == 1) * 0.99  # Make the one hot encoding on new range (0.01-0.99)

outputData = np.zeros([len(rawOutputData), 10])  # pre-allocating output array
inputData = np.zeros([len(rangedInputData), 28, 28])
for trainSet in range(len(rawOutputData)):  # for each training set
    outputData[trainSet, :] = oneHotEncoding[rawOutputData[trainSet]]  # Storing the one hot encoded outputs
    inputData[trainSet] = np.reshape(rangedInputData[trainSet, :], [28, 28])  # Reshapes for image processing (28x28)

# Show image
#plt.imshow(inputData[0], cmap="gray")
#plt.show()

# Generating Filters
numfilters = 10  # How many filters to apply
sizeFilters = 5  # Size of filters
filters = np.random.rand(10, 5, 5)

# MLP Init
hiddennodes = 100
weights1 = np.random.rand(((inputData.shape[1] - filters.shape[1] + 1)//2)**2*numfilters, hiddennodes)
weights2 = np.random.rand(hiddennodes, 10)

loss = np.zeros([1, 100])
for epoch in range(100):
    t = time.time()
    [Pooled, Stored] = cv.convo(inputData, filters, 2)

    # Neural Network
    networkinput = Pooled.reshape([Pooled.shape[0], Pooled.shape[1] * Pooled.shape[2] * Pooled.shape[3]])
    networkinput = networkinput*(networkinput > 0)

    # Feed Forward
    hidden = np.maximum(np.dot(networkinput, weights1), 0)
    output = np.dot(hidden, weights2)
    softmax = np.exp(output - np.amax(output, 1)[:, np.newaxis]) / np.sum(np.exp(output - np.amax(output, 1)[:, np.newaxis]), 1)[:, np.newaxis]

    # Back Propagation
    loss[0, epoch] = sum(sum((softmax - outputData) ** 2))
    dL_dP = 2 * (softmax - outputData)

    softmax_reshape = np.reshape(softmax, (1, -1))
    dL_dP_reshape = np.reshape(dL_dP, (1, -1))
    d_softmax = (softmax_reshape * np.identity(softmax_reshape.size) - softmax_reshape.T @ softmax_reshape)
    dP_dZ3 = (dL_dP_reshape @ d_softmax).reshape([softmax.shape[0], softmax.shape[1]])
    dZ3_dW2 = hidden
    dZ3_dX2 = weights2
    dX2_dZ2 = hidden > 0
    dX2_dW1 = networkinput
    dW2 = np.dot(dZ3_dW2.T, dL_dP * dP_dZ3)
    dW1 = np.dot(dX2_dW1.T, np.dot(dL_dP * dP_dZ3, dZ3_dX2.T) * dX2_dZ2)

    weights1 += dW1*lr
    weights2 += dW2*lr

    dZ2_dX1 = weights1
    dinput = np.dot(np.dot(dL_dP * dP_dZ3, dZ3_dX2.T) * dX2_dZ2, dZ2_dX1.T)
    # Reshape dinput
    dinput_reshape = np.reshape(dinput, Pooled.shape)
    d_act = Pooled > 0
    filters = cv.rconvo(inputData, filters, dinput_reshape*d_act, Stored, lr)

    elapsed = time.time() - t
    print('Epoch: ', epoch, '   Loss: ', round(loss[0, epoch], 2), '   Time: ', round(elapsed, 2))
plt.plot(np.arange(1, loss.size+1), loss.T)
plt.show()
print()
