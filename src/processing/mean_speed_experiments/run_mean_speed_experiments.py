#%%
import sys
#from charsiu.src import models
from charsiu.src.Charsiu import Wav2Vec2ForFrameClassification, CharsiuPreprocessor_en, charsiu_forced_aligner, charsiu_chain_attention_aligner, charsiu_chain_forced_aligner, charsiu_predictive_aligner
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
#%%

#%% Load TIMIT
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Load the Dataframes in the correct directory


record_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_test.csv')
record_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_train.csv')
sample_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_train.csv')
sample_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_test.csv')

#%% -------------------------- GENERATE FEATURES -----------------------------

# Get phonogram features of N_SAMPLES samples in the training set

#phonogram_features_TRAIN = phonograms_to_features(SAMPLE_IDs_TRAIN, train = True)
#phonogram_features_TEST = phonograms_to_features(SAMPLE_IDs_TEST, train = False)
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


#%% ================== CHARSIU PRED ALIGMENT ================================


audio = TIMIT_train[0]
audio_path = audio['file']
charsiu_pred = charsiu_predictive_aligner(aligner='charsiu/en_w2v2_fc_10ms')
aligment = charsiu_pred.align(audio=audio_path)


#%% Generate the data (optional if there is no data)


#charsiu_pred_aligment_test = pd.read_csv('../tesis_speechRate/src/processing/mean_speed_experiments/data/charsiu_<charsiu.src.Charsiu.charsiu_predictive_aligner object at 0x7dc87eec1f90>_test.csv')
#charsiu_pred_aligment_train = pd.read_csv('../tesis_speechRate/src/processing/mean_speed_experiments/data/charsiu_<charsiu.src.Charsiu.charsiu_predictive_aligner object at 0x7dc87eec1f90>_train.csv')



# %% Open the file
charsiu_pred_aligment_test = pd.read_csv('../tesis_speechRate/src/processing/mean_speed_experiments/data/charsiu_<charsiu.src.Charsiu.charsiu_predictive_aligner object at 0x7dc87eec1f90>_test.csv')
charsiu_pred_aligment_train = pd.read_csv('../tesis_speechRate/src/processing/mean_speed_experiments/data/charsiu_<charsiu.src.Charsiu.charsiu_predictive_aligner object at 0x7dc87eec1f90>_train.csv')
# %% change the column phonemes to utterance
charsiu_pred_aligment_train.rename(columns={'phoneme': 'utterance'}, inplace=True)
charsiu_pred_aligment_test.rename(columns={'phoneme': 'utterance'}, inplace=True)

#%% Compute the mean_speed features


charsiu_pred_aligment_train['duration_s'] = charsiu_pred_aligment_train['stop'] - charsiu_pred_aligment_train['start']
charsiu_df_by_sample_train = ut.TIMIT_df_by_sample_phones(charsiu_pred_aligment_train)
#%%
# Each column of charsiu_df_by_sample_train add the name charsiu_pred_aligment 
#charsiu_df_by_sample_train.columns = ['charsiu_pred_aligment_' + str(col) for col in charsiu_df_by_sample_train.columns]

# Merge with X_TRAIN
df_X_TRAIN = pd.merge(df_X_TRAIN, charsiu_df_by_sample_train, left_index=True, right_index=True)





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
# Means
B_mean = ['mean_phone_' + str(i) for i in range(2, 40)]
B_mean_delta = ['mean_delta_phone_' + str(i) for i in range(2, 40)]
B_mean_d_delta = ['mean_d_delta_phone_' + str(i) for i in range(2, 40)]

# Stds
B_std = ['std_phone_' + str(i) for i in range(2, 40)]
B_std_delta = ['std_delta_phone_' + str(i) for i in range(2, 40)]   
B_std_d_delta = ['std_d_delta_phone_' + str(i) for i in range(2, 40)]

# Abs
B_mean_abs = ['mean_abs_phone_' + str(i) for i in range(2, 40)]
B_mean_abs_delta = ['mean_abs_delta_phone_' + str(i) for i in range(2, 40)]
B_mean_abs_d_delta = ['mean_abs_d_delta_phone_' + str(i) for i in range(2, 40)]


B_softmax = ['feature_softmax_' + str(i) for i in range(2, 40)]
B_mean_softmax = ['mean_feature_softmax_' + str(i) for i in range(2, 40)]



B = B_mean + B_std + B_mean_delta + B_std_delta + B_mean_d_delta + B_std_d_delta 

B_abs = B_mean_abs + B_mean_abs_delta + B_mean_abs_d_delta

C = B_mean_softmax

D = ['greedy_feature']

BC = B + C

B_ABS_C = B_abs + C

F = ['charsiu_pred_aligment_mean_speed_wpau_v1']

#%% NEW FEATURES
#G = ['mean_how_many_phones_arFgMax']
#H = ['mean_how_many_probables_phones']
N = 1


mean_phone = df_TRAIN.filter(regex='^mean_phone_*')
#%% Metric 1
y_TRAIN = df_TRAIN['mean_speed_wpau_v1']
y_VAL = df_VAL['mean_speed_wpau_v1']

features = [A,B,C,D,BC, F]
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
#%%


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
plt.xticks(np.arange(len(features)), ['A, dim:' + str(len(A)),'B, dim:'+ str(len(B)),'C, dim:'+ str(len(C)), 'D, dim:'+ str(len(D)), 'BC, dim:'+ str(len(BC)), 'F, dim:'+ str(len(F))]
           , rotation=0)

# Add in this plot the name of each feature
plt.title('Prediction of the speech rate')
plt.xlabel('Groups of features')
plt.ylabel('Mean R2 - '+ str(N) + ' iterations')
plt.legend()
plt.savefig('a_b_c_d_barplot.png')
plt.show()

# %%
