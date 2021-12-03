import os
import numpy as np
import pandas as pd
import librosa.display
import matplotlib.pyplot as plt

audio_fpath1 = "C:\\Users\\schmi\\OneDrive\\Desktop\\School\\MA 305\\MA305_Project\\Audacity Files\\Planets Wav\\Simon"
folders = os.listdir(audio_fpath1)
print("Types of inputs: ", len(folders))

allLabeled = []
foldernum = -1
minRowSize = np.zeros(len(folders))
minColSize = np.zeros(len(folders))
for folder in folders:
    foldernum += 1
    audio_fpath2 = audio_fpath1 + '\\' + folder
    audio_clips = os.listdir(audio_fpath2)
    allRowSize = np.zeros(len(audio_clips))
    allColSize = np.zeros(len(audio_clips))

    filenum = -1
    for file in audio_clips:
        filenum += 1
        x, sr = librosa.load(audio_fpath2+"\\"+file, sr=44100)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        allRowSize[filenum] = Xdb.shape[0]
        allColSize[filenum] = Xdb.shape[1]

    minRowSize[foldernum] = int(min(allRowSize))
    minColSize[foldernum] = int(min(allColSize))
minRowSize = int(min(minRowSize))
minColSize = int(min(minColSize))
minColSize = 65

foldernum = -1
for folder in folders:
    foldernum += 1
    audio_fpath2 = audio_fpath1 + '\\' + folder
    audio_clips = os.listdir(audio_fpath2)
    print("No. of files in folder \"", folder, "\": ", len(audio_clips))
    allSize = np.zeros(len(audio_clips))

    filenum = -1
    allData = np.zeros([minRowSize*minColSize, len(audio_clips)])
    for file in audio_clips:
        filenum += 1
        x, sr = librosa.load(audio_fpath2+"\\"+file, sr=44100)
        X = librosa.stft(x)
        Xdb = librosa.amplitude_to_db(abs(X))
        allData[:, filenum] = np.real(Xdb[0:minRowSize, 0:minColSize].reshape(1, -1))

    labels = np.ones([1, len(audio_clips)]) * foldernum
    labeled = np.append(labels, allData, 0)
    np.random.shuffle(labeled.T)

    if foldernum == 0:
        trainData = labeled[:, 0:40]
        testData = labeled[:, 40:50]
    else:
        trainData = np.append(trainData, labeled[:, 0:40], 1)
        testData = np.append(testData, labeled[:, 40:50], 1)

pd.DataFrame(trainData).to_csv('SimonPlanetsTraining.csv', index=False, header=False)
pd.DataFrame(testData).to_csv('SimonPlanetsTesting.csv', index=False, header=False)

#plt.figure(figsize=(14, 5))
#librosa.display.waveplot(x, sr=sr)

#plt.figure(figsize=(14, 5))
#librosa.display.specshow(Xdb, sr=sr, x_axis='time', y_axis='log')
#plt.colorbar()
