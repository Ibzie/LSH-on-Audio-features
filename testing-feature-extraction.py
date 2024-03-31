import librosa
#import librosa.display
import os
#import IPython.display as ipd
#import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def flatten(l):               #function taken from stackoverflow
    return [item for sublist in l for item in sublist]

def extract_mfccs(file, df, new_df):
    #code taken from youtube: "https://www.youtube.com/watch?v=WJI-17MNpdE&ab_channel=ValerioVelardo-TheSoundofAI"
    signal, sr = librosa.load(file)
    mfccs = librosa.feature.mfcc(signal, n_mfcc=5, sr=sr)           #extracting mfccs

    ##visualizing mfcc
    #plt.figure(figsize=(15,10))
    #librosa.display.specshow(mfccs, x_axis="time", sr=sr)
    #plt.colorbar(format="%+2f")
    ##end of code taken from youtube

    mfccs = flatten(mfccs)

    if (new_df == True):
        df = pd.DataFrame()           #creating a new dataframe
        temp = np.zeros(len(mfccs))
        temp = np.vstack((mfccs, temp))
        df = pd.DataFrame(temp)
        df = df.drop(index=1)       #removing the zeros row from the dataframe
        return df

    else:
        if (len(mfccs) != len(df.columns)):     #normalizing row size
            mfccs = mfccs[:len(df.columns)]

        df.loc[len(df)] = mfccs            #adding the new file's mfccs to the df
        return df


folder = "001"

new = True
mfccs_df = pd.DataFrame()
directory = "A1_Audio\AudioFiles\\031"

for subdir, dirs, files in os.walk(directory):
    print(subdir)
    for file in files:
        if (new == True):
            mfccs_df = extract_mfccs(subdir + "/" + file, mfccs_df, new)
            new = False
        else:
            mfccs_df = extract_mfccs(subdir + "/" + file, mfccs_df, new)

print(mfccs_df)
