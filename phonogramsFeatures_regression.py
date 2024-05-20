#%% #import tables_speechRate as my_tables

#%% TODO 
# 1. Identify the silence and call it 'sil' - OK
# 1.1 Do the necesary changes - OK
# 2. Do the Pablo's features - OK
# 3 Split train and test correctly


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
import seaborn as sns
import ast  # For safely evaluating strings containing Python literals

# ---------------------------- LOAD DATASET -----------------------------------
#%% Load TIMIT
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Load the Dataframes in the correct directory


record_phone_test = pd.read_csv('../tesis_speechRate/TIMIT_df_by_record_phone_test.csv')
record_phone_train = pd.read_csv('../tesis_speechRate/TIMIT_df_by_record_phone_train.csv')
sample_phone_train = pd.read_csv('../tesis_speechRate/TIMIT_df_by_sample_train.csv')
sample_phone_test = pd.read_csv('../tesis_speechRate/TIMIT_df_by_sample_test.csv')


#%% Load any CHARSIU phonogram generated by Get phonograms
#SAMPLE_ID = SAMPLE_IDs_TEST[0]
#phonogram = pd.read_csv('../tesis_speechRate/data_phonograms/CHARSIU/Test/'+SAMPLE_ID+'.csv')
#phonogram = phonogram.to_numpy()
#%% ---------------------------------------------------------------------------



# %% -------------------- FORCED ALIGNMENT -----------------------------------


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
#%%
phonemes_index = np.arange(0,42)
phonemes = [charsiu.charsiu_processor.mapping_id2phone(int(i)) for i in phonemes_index]
print(phonemes)

#%%
# Example: Forced alignment of an audio file

# Audio by my own: 
audio_file = '../tesis_speechRate/audios_tomas/she_has_your.wav'  # Replace with the path to your audio file
waveform, sample_rate = librosa.load(audio_file, sr=None)
# Play audio


x = torch.tensor(np.array([waveform]).astype(np.float32))
with torch.no_grad():
    y = modelo(x)
    y = modelo(x).logits
    y_softmax = torch.softmax(y, dim=2)

y = y.numpy()[0].T
y_softmax = y_softmax.numpy()[0].T

#y_softmax[0,:] = 1
plt.figure(figsize=(10, 7))
plt.pcolor(y_softmax)
plt.yticks(np.arange(0.5, 42.5, 1), phonemes)
plt.title('Phonogram')
plt.colorbar()
plt.show()


# ----------------------------------------------------------------------------

#%% With softmax
plt.figure(figsize=(10, 7))
plt.pcolor(y_softmax)
plt.yticks(np.arange(0.5, 42.5, 1), phonemes)
plt.title('Phonogram')
plt.colorbar()
plt.show()


#%% ======================= FUNCTIONS ==================================
def phonograms_to_features(sample_IDs, train = True):
    print('==============GETTING PHONOGRAMS FEATURES================')
    features = pd.DataFrame()
    for i in range(len(sample_IDs)):

        features = pd.concat([features, ut.phonogram_to_features(sample_ID=sample_IDs[i], train=train)], ignore_index=True)

        if i % 10 == 0:
            print('SAMPLE ', i, ' OF ', len(sample_IDs))
            print('-------------------------------------------------')
    #save features
    if train:
        features.to_csv('../tesis_speechRate/src/processing/data_phonograms/data_features/Train/features_train.csv', index=False)
    else:
        features.to_csv('../tesis_speechRate/src/processing/data_phonograms/data_features/Test/features_test.csv', index=False)

    print('=================FINISHED===================')        
    return features

#def batch_phonograms_to_features(sample_IDs, train = True, batch_size = 10):
#    print('==============GETTING BATCH PHONOGRAMS FEATURES================')
#    features = pd.DataFrame()
#    for i in range(len(sample_IDs)):
           



#%% -------------------------- GENERATE PHONOGRAMS -----------------------------
# OBS: This process takes a long time. It save the phonograms as csv files in the data_phonograms folder 


N_TRAIN = len(TIMIT_train)
N_TEST = len(TIMIT_test)

#get_phonograms(TIMIT_train, modelo, 10, train=True)



# =======Get phonograms TEST============= UNCOMMENT TO GET PHONOGRAMS

#get_phonograms(TIMIT_test, modelo, N_TEST, train=False)


#%% ---------------------------- GLOBAL VARIABLES --------------------------------

SAMPLE_IDs_TRAIN = ut.get_sample_IDs(TIMIT_train, N_TRAIN)
SAMPLE_IDs_TEST = ut.get_sample_IDs(TIMIT_test, N_TEST)


# ------------------------------- TEST FUNCTIONS --------------------------------------
#%% Test the functions
SAMPLE_ID = SAMPLE_IDs_TRAIN[0]
# read the phonogram
phonogram = pd.read_csv('../tesis_speechRate/src/processing/data_phonograms/CHARSIU/Train/'+SAMPLE_ID+'.csv')

plt.figure(figsize=(10, 7))
plt.pcolor(phonogram)
plt.yticks(np.arange(0.5, 42.5, 1), phonemes)
plt.title('Phonogram')
plt.colorbar()
plt.show()

phonogram = phonogram.to_numpy()
phonogram_softmax = ut.softmax_phonogram(phonogram)
phonogram_softmax = phonogram_softmax > 0.5
plt.figure(figsize=(10, 7))
plt.pcolor(phonogram_softmax)
plt.yticks(np.arange(0.5, 42.5, 1), phonemes)
plt.colorbar()
plt.title('Phonogram')

#%%

#how_many_phones_since_t(phonogram)
#print(how_many_phones_since_t(phonogram)[1])
#mean_phones_arg_max(phonogram)
ut.how_many_probables_phones(phonogram)[0]

#%% -------------------------- GENERATE FEATURES -----------------------------

# Get phonogram features of N_SAMPLES samples in the training set

phonogram_features_TRAIN = phonograms_to_features(SAMPLE_IDs_TRAIN, train = True)
phonogram_features_TEST = phonograms_to_features(SAMPLE_IDs_TEST, train = False)
#%%
# READ FEATURES 
phonogram_features_TRAIN = pd.read_csv('../tesis_speechRate/src/processing/data_phonograms/data_features/Train/features_train.csv')
phonogram_features_TEST = pd.read_csv('../tesis_speechRate/src/processing/data_phonograms/data_features/Test/features_test.csv')

#%%
phonogram_features_TRAIN.set_index('sample_id', inplace=True)
phonogram_features_TEST.set_index('sample_id', inplace=True)


# Merge the phonogram features with the sample_phone dataframes
sample_phone_train.set_index('sample_id', inplace=True)
sample_phone_test.set_index('sample_id', inplace=True)
#%%
df_X_TRAIN = pd.merge(phonogram_features_TRAIN, sample_phone_train, left_index=True, right_index=True)
#%% -------------------------- SPLITTING -------------------------------------
speaker_id = df_X_TRAIN['speaker_id'].unique()
n_speakers = len(speaker_id)

# 80% Train - 20% Val
n_train = round(0.8*n_speakers)
n_val = n_speakers - n_train
# Choose randomly
random.shuffle(speaker_id)
speaker_id_train = speaker_id[:n_train]
speaker_id_val = speaker_id[n_train:]
#%%
# Filter the speaker_id_train
df_TRAIN = df_X_TRAIN[df_X_TRAIN['speaker_id'].isin(speaker_id_train)]
df_VAL = df_X_TRAIN[df_X_TRAIN['speaker_id'].isin(speaker_id_val)]


# %% TRAIN SET
X_TRAIN  = df_TRAIN.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_TRAIN = df_TRAIN['mean_speed_wpau_v1']
X_VAL = df_VAL.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_VAL = df_VAL['mean_speed_wpau_v1']
# %% =================== FEATURES SELECTION ===================================

A = ['all_mean_phonogram', 
     'all_mean_delta', 
     'all_mean_d_delta', 
     'all_std_phonogram',
     'all_mean_abs_phonogram',
     'all_mean_abs_delta',
     'all_mean_abs_d_delta'] 



#%% ALl PHONES FEATURES
B = []
B_mean = ['mean_phone_' + str(i) for i in range(2, 40)]
B_std = ['std_phone_' + str(i) for i in range(2, 40)]
B_mean_delta = ['mean_delta_phone_' + str(i) for i in range(2, 40)]
B_std_delta = ['std_delta_phone_' + str(i) for i in range(2, 40)]   
B_mean_d_delta = ['mean_d_delta_phone_' + str(i) for i in range(2, 40)]
B_std_d_delta = ['std_d_delta_phone_' + str(i) for i in range(2, 40)]
B_abs = ['mean_abs_phone_' + str(i) for i in range(2, 40)]
B_abs_delta = ['mean_abs_delta_phone_' + str(i) for i in range(2, 40)]
B_abs_d_delta = ['mean_abs_d_delta_phone_' + str(i) for i in range(2, 40)]
B_softmax = ['feature_softmax_' + str(i) for i in range(2, 40)]
B_mean_softmax = ['mean_feature_softmax_' + str(i) for i in range(2, 40)]



B = B_mean + B_std + B_mean_delta + B_std_delta + B_mean_d_delta + B_std_d_delta + B_abs + B_abs_delta + B_abs_d_delta + B_softmax + B_mean_softmax

C = B_mean_softmax

D = ['greedy_feature']

#%% NEW FEATURES
#G = ['mean_how_many_phones_arFgMax']
#H = ['mean_how_many_probables_phones']
N = 10


mean_phone = df_TRAIN.filter(regex='^mean_phone_*')
#%% Metric 1
y_TRAIN = df_TRAIN['mean_speed_wpau_v1']
y_VAL = df_VAL['mean_speed_wpau_v1']

features = [A,B, C, D]
MSE_features_wpau = np.zeros(len(features))
scores_wpau = np.zeros(len(features))
for j in range(N):
    for i in range(len(features)):
        print('Features:', features[i])
        X_TRAIN_fi = X_TRAIN[features[i]]
        X_VAL_fi = X_VAL[features[i]]
        
        # Regression
        positive=True
        model = linear_model.LinearRegression(positive=positive)
        model.fit(X_TRAIN_fi, y_TRAIN)
        y_pred = model.predict(X_VAL_fi)
        MSE_features_wpau[i] += mean_squared_error(y_VAL, y_pred)
        scores_wpau[i] += model.score(X_VAL_fi, y_VAL)

mean_score_features_wpau = scores_wpau/N
mean_MSE_features_wpau = MSE_features_wpau/N

#%% Metric 2
y_TRAIN = df_TRAIN['mean_speed_wopau_v1']
y_VAL = df_VAL['mean_speed_wopau_v1']
MSE_features_wopau = np.zeros(len(features))
scores_wopau = np.zeros(len(features))
for j in range(N):
    for i in range(len(features)):
        print('Features:', features[i])
        X_TRAIN_fi = X_TRAIN[features[i]]
        X_VAL_fi = X_VAL[features[i]]
        
        # Regression
        positive=True
        model = linear_model.LinearRegression(positive=positive)
        model.fit(X_TRAIN_fi, y_TRAIN)
        y_pred = model.predict(X_VAL_fi)
        MSE_features_wopau[i] += mean_squared_error(y_VAL, y_pred)
        scores_wopau[i] += model.score(X_VAL_fi, y_VAL)

mean_score_features_wopau = scores_wopau/N
mean_MSE_features_wopau = MSE_features_wopau/N


#%%
# Do barplot of the features with MSE add one next to the other
x = np.arange(len(features))
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_score_features_wpau, width, label='With pauses v1')
rects2 = ax.bar(x + width/2, mean_score_features_wopau, width, label='With out pauses v1')
plt.xticks(np.arange(len(features)), ['A, dim:' + str(len(A)),'B, dim:'+ str(len(B)),'C, dim:'+ str(len(C)), 'D, dim:'+ str(len(D))]
           , rotation=0)

# Add in this plot the name of each feature
plt.title('Prediction of the speech rate')
plt.xlabel('Groups of features')
plt.ylabel('Mean R2 - '+ str(N) + ' iterations')
plt.legend()
plt.savefig('a_b_c_d_barplot.png')
plt.show()


# %% Correlation Matrix

ALL_FEATURES = A + B + C + D
X_TRAIN_ALL = X_TRAIN[ALL_FEATURES]
sns.heatmap(X_TRAIN_ALL.corr())
plt.title('Correlation Matrix of the features')




# %% 
sns.heatmap(mean_phone.corr())
# %%


X_TRAIN
# %% 
sns.pairplot(pd.concat([X_TRAIN.filtermean_speed_wopau(regex='all_*'),y_TRAIN],axis=1), hue="mean_speed_wopau", palette="husl")


# %% Generate Features per batch
