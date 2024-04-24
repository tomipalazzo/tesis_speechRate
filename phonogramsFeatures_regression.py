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
#import src.tables_speechRate as my_tables
import src.utils as ut
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#%% Load the Dataframes of each record
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%%

TIMIT_df_by_record_phone_test = pd.read_csv('TIMIT_df_by_record_phone_test.csv')
TIMIT_df_by_record_phone_train = pd.read_csv('TIMIT_df_by_record_phone_train.csv')
TIMIT_df_by_sample_train = pd.read_csv('TIMIT_df_by_sample_train.csv')
TIMIT_df_by_sample_test = pd.read_csv('TIMIT_df_by_sample_test.csv')

#%%
TIMIT_df_by_sample_train
#%% Global Variables
N_SAMPLES = 100

#%%
# Load the dataset
sample_phone_train = TIMIT_df_by_record_phone_train

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
TIMIT_train = TIMIT_train
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

TIMIT_df_by_sample_train 
# %%
TIMIT_df_by_sample_train.set_index('sample_id', inplace=True)
# %%
df_X = pd.merge(phonograms_features_df, TIMIT_df_by_sample_train, left_index=True, right_index=True)
# %%
X = df_X.drop(columns=['mean_speed_wpau', 'mean_speed_wopau', 'duration_wpau'])
y = df_X['mean_speed_wpau']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = linear_model.LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mean_squared_error(y_test, y_pred)

#%%
y_train_pred = model.predict(X_train)
mean_squared_error(y_train, y_train_pred)
# R2 score
model.score(X_train, y_train)
model.score(X_test, y_test)

#%% PCA 
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
# 95% of the variance
np.where(np.cumsum(pca.explained_variance_ratio_) < 0.95)


pca = PCA(n_components=1)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
model = linear_model.LinearRegression()
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
mean_squared_error(y_test, y_pred)
model.score(X_test_pca, y_test)
print(model.score(X_train_pca, y_train)
)
plt.figure()
plt.scatter(X_train_pca, y_train)

# %%
df_X.corr()
# %%
import seaborn as sns
sns.heatmap(df_X.corr())
# %%
