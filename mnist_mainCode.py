import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import convolution as cv
import time

tTill = 10  # How many of inputs to run through
lr = 1e-7
poolSize = 2
numfilters = 10  # How many filters to apply
sizeFilters = 5  # Size of filters
hiddennodes = 100
epochs = 100
train = False  # Assumes evaluation is wanted if set to false
reset = False  # Determines whether weights get reset

if train:
    trainData = np.array(pd.read_csv("mnist_train.csv"))
    np.random.shuffle(trainData)
    rawOutputData = trainData[0:tTill, 0].astype(int)  # Need this as one-hot encoding
    rawInputData = trainData[0:tTill, 1:trainData.shape[1] + 1]  # Need this on a range of 0.01-1, currently 0-255
else:
    testData = np.array(pd.read_csv("mnist_test.csv"))
    np.random.shuffle(testData)
    rawOutputData = testData[0:tTill, 0].astype(int)  # Need this as one-hot encoding
    rawInputData = testData[0:tTill, 1:testData.shape[1] + 1]  # Need this on a range of 0.01-1, currently 0-255\


rangedInputData = rawInputData * 0.99 / 255 + 0.01  # Puts the input data on a new range (0.01-1)
diagArray = np.diagflat([np.ones(10)])  # Define the original one hot encoding
oneHotEncoding = (diagArray == 0) * 0.01 + (diagArray == 1) * 0.99  # Make the one hot encoding on new range (0.01-0.99)

outputData = np.zeros([len(rawOutputData), 10])  # pre-allocating output array
inputData = np.zeros([len(rangedInputData), 28, 28])
for Set in range(len(rawOutputData)):  # for each training set
    outputData[Set, :] = oneHotEncoding[rawOutputData[Set]]  # Storing the one hot encoded outputs
    inputData[Set] = np.reshape(rangedInputData[Set, :], [28, 28])  # Reshapes for image processing (28x28)

# Show image
#plt.imshow(inputData[0], cmap="gray")
#plt.show()

if train:
    # MLP Init
    if reset:
        filters = np.random.rand(10, 5, 5)
        weights1 = np.random.rand(filters.shape[0] * (inputData.shape[1] - filters.shape[1] + 1) // poolSize * (
                inputData.shape[2] - filters.shape[2] + 1) // poolSize, hiddennodes)
        weights2 = np.random.rand(hiddennodes, outputData.shape[1])
    else:
        # Initializing Filters
        filters = np.zeros([numfilters, sizeFilters, sizeFilters])

        # Open weights and filters
        weights1DF = pd.read_csv('mnistWeights1.csv')
        weights2DF = pd.read_csv('mnistWeights2.csv')

        reader = pd.ExcelFile('mnistFilters.xlsx', engine='openpyxl')
        for n in range(numfilters):
            curFilter = pd.read_excel(reader, sheet_name="Filter " + str(n))
            filters[n, :, :] = np.array(curFilter)[:, 1:np.array(curFilter).shape[1]]
        reader.close()

        weights1 = np.array(weights1DF)[:, 1:np.array(weights1DF).shape[1]]
        weights2 = np.array(weights2DF)[:, 1:np.array(weights2DF).shape[1]]

    loss = np.zeros([1, epochs])
    for epoch in range(epochs):
        t = time.time()
        convolution = cv.convo(inputData, filters)
        [Pooled, Stored] = cv.pooling(convolution, inputData, filters, poolSize)

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
        newFilters = cv.rconvo(inputData, filters, dinput_reshape * d_act, Stored)

        filterChange = np.zeros_like(filters)
        for f in range(filters.shape[0]):
            for i in range(inputData.shape[0]):
                filterChange[f] = filterChange[f] + newFilters[i, f]

            filters[f] = filters[f] + filterChange[f] * lr

        # End training step, report stats
        elapsed = time.time() - t
        print('Epoch: ', epoch, '   Loss: ', round(loss[0, epoch], 2), '   Time: ', round(elapsed, 2))

    # Save weights and filters
    pd.DataFrame(weights1).to_csv('mnistWeights1.csv')
    pd.DataFrame(weights2).to_csv('mnistWeights2.csv')

    # Save Filters
    writer = pd.ExcelWriter('mnistFilters.xlsx', engine='openpyxl')
    for n in range(numfilters):
        pd.DataFrame(filters[0]).to_excel(writer, sheet_name="Filter " + str(n))
    writer.save()
    writer.close()

    plt.plot(np.arange(1, loss.size+1), loss.T)
    plt.show()
    print()
else:
    # Initializing Filters
    filters = np.zeros([numfilters, sizeFilters, sizeFilters])

    # Open weights and filters
    weights1DF = pd.read_csv('mnistWeights1.csv')
    weights2DF = pd.read_csv('mnistWeights2.csv')

    reader = pd.ExcelFile('mnistFilters.xlsx', engine='openpyxl')
    for n in range(numfilters):
        curFilter = pd.read_excel(reader, sheet_name="Filter " + str(n))
        filters[n, :, :] = np.array(curFilter)[:, 1:np.array(curFilter).shape[1]]
    reader.close()

    weights1 = np.array(weights1DF)[:, 1:np.array(weights1DF).shape[1]]
    weights2 = np.array(weights2DF)[:, 1:np.array(weights2DF).shape[1]]

    t = time.time()
    convolution = cv.convo(inputData, filters)
    [Pooled, Stored] = cv.pooling(convolution, inputData, filters, poolSize)

    # Neural Network
    networkinput = Pooled.reshape([Pooled.shape[0], Pooled.shape[1] * Pooled.shape[2] * Pooled.shape[3]])
    networkinput = networkinput * (networkinput > 0)

    # Feed Forward
    hidden = np.maximum(np.dot(networkinput, weights1), 0)
    output = np.dot(hidden, weights2)
    softmax = np.exp(output - np.amax(output, 1)[:, np.newaxis]) / np.sum(
        np.exp(output - np.amax(output, 1)[:, np.newaxis]), 1)[:, np.newaxis]

    # Loss
    loss = sum(sum((softmax - outputData) ** 2))

    # End training step, report stats
    elapsed = time.time() - t
    print('   Loss: ', round(loss, 2), '   Time: ', round(elapsed, 2))
    print()

