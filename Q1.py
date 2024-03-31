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


#UNOCMMENT FOR PREPROCESSING THE AUDIO
#length of list is 6465|6455
#For Sample1 of audio Features


directory = os.path.join('A1_Audio','AudioFiles')
for subdir, dirs, files in os.walk(directory):
    for file in files:
        print(os.path.join(directory,file))
        if (new == True):
            Audio_Features = extract_mfccs(subdir + "/" + file, Audio_Features, new)
            new = False
        else:
            Audio_Features = extract_mfccs(subdir + "/" + file, Audio_Features, new)

Audio_Features.to_csv('AudioFeatures1.csv')

#FOR DIRECTLY CALLING PREPROCESSED DATA
# dataframe = pd.read_csv('AudioFeatures1.csv',index_col=0)
#print(dataframe)

'''
The following classes used are from https://santhoshhari.github.io/Locality-Sensitive-Hashing/
The concept of cosine similarity over multiple hash tables using projections was used.
Minhashing proved to be inefficent compared to applying Multi-Hash Cossine Similarity
'''


#Getting Random Projection B Vectors with n = 2
#SAMPLE TESTING
# vec1 = np.array(dataframe.iloc[0])
# print("Vector1:",vec1, vec1.shape)
#
# vec2 = np.array(dataframe.iloc[1])
# print("Vector2:",vec2, vec2.shape)
#
# projection = np.random.randn(6, 6456)
# print("Projectinos:",projection, projection.shape)
#
# B_vector_list = []
# for i in range (0, 2234):
#     temp_vector = np.array(dataframe.iloc[i])
#     res = ''.join((np.dot(temp_vector, projection.T) > 0).astype(int).astype('str'))
#     print(res)
#     B_vector_list.append(res)
# print("\n\n")
# print(B_vector_list)


def cosine_similarity(v1, v2):
    return np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

class HashTable:
    def __init__(self, hash_size, inp_dimensions):
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_table = dict()
        self.projections = np.random.randn(self.hash_size, inp_dimensions)

    def generate_hash(self, inp_vector):
        bools = (np.dot(inp_vector, self.projections.T) > 0).astype('int')
        return ''.join(bools.astype('str'))

    def __setitem__(self, inp_vec, label):
        hash_value = self.generate_hash(inp_vec)
        self.hash_table[hash_value] = self.hash_table \
                                          .get(hash_value, list()) + [label]

    def __getitem__(self, inp_vec):
        hash_value = self.generate_hash(inp_vec)
        return self.hash_table.get(hash_value, [])


hash_table = HashTable(hash_size=16, inp_dimensions=6455)

class LSH:
    def __init__(self, num_tables, hash_size, inp_dimensions):
        self.num_tables = num_tables
        self.hash_size = hash_size
        self.inp_dimensions = inp_dimensions
        self.hash_tables = list()
        for i in range(self.num_tables):
            self.hash_tables.append(HashTable(self.hash_size, self.inp_dimensions))

    def __setitem__(self, inp_vec, label):
        for table in self.hash_tables:
            table[inp_vec] = label

    def __getitem__(self, inp_vec):
        results = list()
        for table in self.hash_tables:
            results.extend(table[inp_vec])
        return list(set(results))

#changing the first parameter of LSH \/ will increase matches but also increase process time
Sys = LSH(1, 16, 6455)

#loading the LSH
for i in range(0, 2234):
    # temp_hash = hash_table.generate_hash(np.array(dataframe.iloc[i]))
    # print(temp_hash)
    Sys.__setitem__(np.array(dataframe.iloc[i]),str(i))


#getting pairs with Audio Label
# for i in range(0, 2234):
#     res = Sys.__getitem__(np.array(dataframe.iloc[i]))
#     print(i)
    #print('For audio number:',i,' the list is: ',res,'\n\n\n')
min = 1
Min_name = 1
max = 0
Max_name = 0
#length of dataframe is 2235 width is 6455


for j in range(0,2234):
    res = Sys.__getitem__(np.array(dataframe.iloc[j]))
    for i in res:
        V1 = np.array(dataframe[str(i)])
        for x in res:
            if i == x:
                continue
            v2 = np.array(dataframe[str(x)])
            temp_result = cosine_similarity(V1, v2)
            if max < temp_result and i != x:
                max = temp_result
                Max_name = x
            if min > temp_result and i != x:
                min = temp_result
                Min_name = x
            # if temp_result >= 0.9:
            #     print(i," exactly matches for the audio vector: ",x)
            # if temp_result <=0.2:
            #     print(i, " has nothing in similar with audio ", x)
        print("for aduio", i)
        print(min, Min_name)
        print(max, Max_name)
        min = 1
        Min_name = 1
        max = 0
        Max_name = 0
    print("##########################################")
    print("END OF LIST OF PAIRED VECTORS NUMBER ", j)
    print("##########################################")
#print(res)
#print(dataframe)


