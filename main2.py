import librosa
import os
import numpy as np
import pandas as pd
import warnings

def flatten(l):               #function taken from stackoverflow
    return [item for sublist in l for item in sublist]

def extract_mfccs(file, df, new_df):

    #code taken from youtube: "https://www.youtube.com/watch?v=WJI-17MNpdE&ab_channel=ValerioVelardo-TheSoundofAI"
    signal, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(signal, n_mfcc=5, sr=sr)           #extracting mfccs

    ##end of code taken from youtube

    mfccs = flatten(mfccs)
    mfccs = mfccs[:len(df.columns)]     #normalizing row size

    df.loc[len(df)] = mfccs            #adding the new file's mfccs to the df
    return df


Audio_Column_List = []
for i in range (0,6455):
    Audio_Column_List.append(i)

new = False
Audio_Features = pd.DataFrame(columns=Audio_Column_List)
mfc = []
count = 0
#print(Audio_Features)
#THis loop works for all audio files
# directory = os.path.join('A1_Audio','AudioFiles')
# for filename in os.listdir(directory):
#     f = os.path.join(directory, filename)
#     # checking if it is a file
#     for x in os.listdir(f):
#         #Audio_List.append(f+x)
#         temp, sr = librosa.load(os.path.join(f, x))
#         mfc = librosa.feature.mfcc(temp, n_mfcc=5, sr=sr)
#         flatten_mfcc = list(np.concatenate(mfc).flat)
#         Audio_Features.append(flatten_mfcc)

#length of list is 6465|6455
#For Sample1 of audio Features

directory = "E:\Projects\BDAA1\A1_Audio\AudioFiles\Sample2"

# directory = os.path.join('A1_Audio','AudioFiles','031')
# for subdir, dirs, files in os.walk(directory):
#     for file in files:
#         if (new == True):
#             Audio_Features = extract_mfccs(subdir + "/" + file, Audio_Features, new)
#             new = False
#         else:
#             Audio_Features = extract_mfccs(subdir + "/" + file, Audio_Features, new)

# Audio_Features.to_csv('AudioFeatures.csv')
dataframe = pd.read_csv('AudioFeatures.csv')

import time
import re
from datasketch import MinHash, MinHashLSHForest

import pyminhash



