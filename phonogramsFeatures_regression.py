#%% #import tables_speechRate as my_tables

import sys
#from charsiu.src import models
from charsiu.src.Charsiu import Wav2Vec2ForFrameClassification, CharsiuPreprocessor_en, charsiu_forced_aligner
import torch 
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from datasets import load_dataset
import pandas as pd
import random
import librosa
import src.tables_speechRate as my_tables
import src.utils as ut
import time


#%% Global Variables
N_SAMPLES = 50

#%%
# Load the dataset
sample_phone_train = my_tables.TIMIT_df_by_record.phone_train

# %%
sample_phone_train
# %% Magic happens here


# Line 1: Instantiate a forced aligner object from the 'charsiu' library using a specified model.
# 'aligner' specifies the model to be used for alignment, likely based on the Wav2Vec 2.0 model trained for frame classification every 10 milliseconds.
charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

# Line 2: Load a pre-trained model from Hugging Face's 'transformers' library.
# This model is likely a fine-tuned version of Wav2Vec 2.0 for the task of frame classification, useful for tasks like forced alignment or phoneme recognition.
modelo = Wav2Vec2ForFrameClassification.from_pretrained("charsiu/en_w2v2_fc_10ms")

# Line 3: Set the model to evaluation mode. This disables training specific behaviors like dropout, 
# ensuring the model's inference behavior is consistent and deterministic.
modelo.eval()

# Line 4: Instantiate a preprocessor for English from the 'charsiu' library.
# This object is likely used to prepare audio data by normalizing or applying necessary transformations 
# before it can be inputted to the model.
procesador = CharsiuPreprocessor_en()

#%% Do a Dataframe with sample_ID and the audio array
TIMIT_train = my_tables.TIMIT_train
TIMIT_train_df = pd.DataFrame(TIMIT_train)
#%%
def get_phonograms(TIMIT_set, model, n_samples = N_SAMPLES):
    t0 = time.time()
    phonograms = []
    for i in range(n_samples):
        sample = TIMIT_set[i]
        audio_data = sample['audio']['array']
        x = torch.tensor(np.array([audio_data]).astype(np.float32))
        with torch.no_grad():
            y = model(x).logits
        y = y.numpy()[0].T
        phonograms.append(y)
    t1 = time.time()
    print('Time to get phonograms: ', t1-t0)
    return phonograms
# %%

# %% Dataframes with the array of the samples (array_id, array)
sample_id = []

for i in range(N_SAMPLES):
    sample = TIMIT_train[i]
    id = sample['dialect_region'] + '_' + sample['speaker_id'] + '_' + sample['id']
    sample_id.append(id)
#%% 

phonograms = get_phonograms(TIMIT_train, modelo)

# %% Dataframe (sample_id, phonogram)
phonograms_df = pd.DataFrame({'sample_id': sample_id, 'phonogram': phonograms})

# %% Phonogram features
def get_phonogram_features(phonogram, sample_id):
    delta = librosa.feature.delta(phonogram)
    d_delta = librosa.feature.delta(phonogram, order=2)
    '''
    for i in range(phonogram.shape[0]):
        features.append({           'sample_id': sample_id,
                                    'mean_phonogram': np.mean(phonogram[i,:]),
                                    'std_phonogram': np.std(phonogram[i,:]),
                                    'mean_delta': np.mean(delta[i,:]),
                                    'std_delta': np.std(delta[i,:]),
                                    'mean_d_delta': np.mean(d_delta[i,:]),
                                    'std_d_delta': np.std(d_delta[i,:]),
                                    'abs_mean_phonogram': np.mean(np.abs(phonogram[i,:])),
                                    'abs_mean_delta': np.mean(np.abs(delta[i,:])),
                                    'abs_mean_d_delta': np.mean(np.abs(d_delta[i,:]))})

    '''
    for i in range(phonogram.shape[0]):
        dic = {'sample_id': sample_id}
        for j in range(1,42+1): # 42 is the number of phonemes
            dic['mean_phonogram_' + str(j)] = np.mean(phonogram[i,:])
            dic['std_phonogram_' + str(j)] = np.std(phonogram[i,:])
            dic['mean_delta_' + str(j)] = np.mean(delta[i,:])
            dic['std_delta_' + str(j)] = np.std(delta[i,:])
            dic['mean_d_delta_' + str(j)] = np.mean(d_delta[i,:])
            dic['std_d_delta_' + str(j)] = np.std(d_delta[i,:])
            dic['abs_mean_phonogram_' + str(j)] = np.mean(np.abs(phonogram[i,:]))
            dic['abs_mean_delta_' + str(j)] = np.mean(np.abs(delta[i,:]))
            dic['abs_mean_d_delta_' + str(j)] = np.mean(np.abs(d_delta[i,:]))

    features = pd.DataFrame(dic, index=[0])
    return features  
    

#%%
phonograms_features = []
for i in range(len(phonograms)):
    phonograms_features.append(get_phonogram_features(phonograms[i], sample_id=sample_id[i]))

# %%
phonograms_features_df = pd.concat(phonograms_features)
phonograms_features_df.set_index('sample_id', inplace=True)
#%%

phonograms_features_df
#%% 

TIMIT_df_by_sample_train = ut.TIMIT_df_by_sample_phones(my_tables.TIMIT_df_by_record.phone_train)
# %%
TIMIT_df_by_sample_train
# %%
df_X = pd.merge(phonograms_features_df, TIMIT_df_by_sample_train, left_index=True, right_index=True)
# %%
X = df_X.drop(columns=['mean_speed_wpau', 'mean_speed_wopau'])
y = df_X['mean_speed_wpau']
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mean_squared_error(y_test, y_pred)

#%%
y2 = df_X['mean_speed_wopau']
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mean_squared_error(y_test, y_pred)


# %%
