import numpy as np
import time


def convo(inputData, filters):

    # Init
    conv = np.zeros([inputData.shape[0], filters.shape[0], inputData.shape[1]-filters.shape[1]+1, inputData.shape[2]-filters.shape[2]+1])

    runtime = 0
    convStart = 0
    convEnd = 0

    for i in range(inputData.shape[0]):
        # Timing
        speed = time.time() - runtime
        left = inputData.shape[0]-i-1
        hour = speed*left/3600
        minute = 60*(hour - int(hour))
        second = 60*(minute - int(minute))

        print("Convoluting... ", i, "/", inputData.shape[0] - 1,
              "     Speed: ", round(convEnd - convStart, 3),
              "     Estimated Time Left: ", int(hour), ':', int(minute), ':', int(second), end="\r")
        runtime = time.time()
        for f in range(filters.shape[0]):

            convStart = time.time()
            # Convolution
            for r in range(inputData.shape[1]-filters.shape[1]+1):
                for c in range(inputData.shape[2]-filters.shape[2]+1):
                    conv[i, f, r, c] = (inputData[i, r:r+filters.shape[1], c:c+filters.shape[2]] * filters[f]).sum()
            convEnd = time.time()

    return conv


def pooling(conv, inputData, filters, poolSize):
    # Init
    Pool = np.zeros([conv.shape[0], filters.shape[0], conv.shape[2]//poolSize, conv.shape[3]//poolSize])
    maxLoc = np.zeros([(inputData.shape[1]-filters.shape[1])//2+1, (inputData.shape[2]-filters.shape[2])//2+1])
    Store = np.zeros_like(conv)
    poolRowDim = conv.shape[2] // poolSize
    poolColDim = conv.shape[3] // poolSize

    runtime = 0
    poolStart = 0
    poolEnd = 0

    # Pooling
    for i in range(conv.shape[0]):
        # Timing
        speed = time.time() - runtime
        left = conv.shape[0] - i - 1
        hour = speed * left / 3600
        minute = 60 * (hour - int(hour))
        second = 60 * (minute - int(minute))

        print("Pooling... ", i, "/", conv.shape[0] - 1,
              "     Speed: ", round(poolEnd - poolStart, 3),
              "     Estimated Time Left: ", int(hour), ':', int(minute), ':', int(second), end="\r")
        runtime = time.time()
        for f in range(conv.shape[1]):
            poolStart = time.time()
            for r in range(poolRowDim):
                for c in range(poolColDim):
                    Pool[i, f, r, c] = np.max(conv[i, f, poolSize*r:poolSize*r+poolSize, poolSize*c:poolSize*c+poolSize])
                    maxLoc[r, c] = np.argmax(conv[i, f, poolSize*r:poolSize*r+poolSize, poolSize*c:poolSize*c+poolSize])
                    pos = np.unravel_index(int(maxLoc[r, c]), conv[i, f, poolSize*r:poolSize*r+poolSize, poolSize*c:poolSize*c+poolSize].shape)
                    Store[i, f, pos[0]+poolSize*r, pos[1]+poolSize*c] = 1
            poolEnd = time.time()

    return Pool, Store


def rconvo(inputData, filters, gradient, Store):

    # Init
    backprop = np.zeros_like(Store)
    newFilters = np.zeros([inputData.shape[0], filters.shape[0], inputData.shape[1]-backprop.shape[2]+1, inputData.shape[2]-backprop.shape[3]+1])
    poolDim = Store.shape[2] // gradient.shape[2]

    runtime = 0
    for i in range(inputData.shape[0]):
        speed = time.time() - runtime
        left = inputData.shape[0] - i - 1
        hour = speed * left / 3600
        minute = 60 * (hour - int(hour))
        second = 60 * (minute - int(minute))
        print("Backpropagating... ", i, "/", inputData.shape[0] - 1,
              "     Estimated Time Left ", int(hour), ':', int(minute), ':', int(second), end="\r")
        runtime = time.time()
        for f in range(filters.shape[0]):
            for r in range(gradient.shape[2]):
                for c in range(gradient.shape[3]):
                    backprop[i, f, r*poolDim:r*poolDim+poolDim, c*poolDim:c*poolDim+poolDim]\
                        = Store[i, f, r*poolDim:r*poolDim+poolDim, c*poolDim:c*poolDim+poolDim] * gradient[i, f, r, c]

            for r in range(inputData.shape[1]-backprop.shape[2]+1):
                for c in range(inputData.shape[2]-backprop.shape[3]+1):
                    newFilters[i, f, r, c] = (inputData[i, r:r + backprop.shape[2], c:c + backprop.shape[3]] * backprop[i, f]).sum()

    return newFilters