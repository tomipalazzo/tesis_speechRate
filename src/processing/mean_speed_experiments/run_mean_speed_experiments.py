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
#%% Merge the phonogram features with the sample_phone dataframes 

df_X_TRAIN = pd.merge(phonogram_features_TRAIN, sample_phone_train, left_index=True, right_index=True)

df_X_TEST = pd.merge(phonogram_features_TEST, sample_phone_test, left_index=True, right_index=True)

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

# TRAIN
charsiu_pred_aligment_train['duration_s'] = charsiu_pred_aligment_train['stop'] - charsiu_pred_aligment_train['start']
charsiu_pred_aligment_train['phone_rate'] = 1/charsiu_pred_aligment_train['duration_s']
# if utterance is '[SIL]' then phone_rate = 0
charsiu_pred_aligment_train['phone_rate'] = charsiu_pred_aligment_train['phone_rate'].where(charsiu_pred_aligment_train['utterance'] != '[SIL]', 0)

# TEST

charsiu_pred_aligment_test['duration_s'] = charsiu_pred_aligment_test['stop'] - charsiu_pred_aligment_test['start']
charsiu_pred_aligment_test['phone_rate'] = 1/charsiu_pred_aligment_test['duration_s']
# if utterance is '[SIL]' then phone_rate = 0
charsiu_pred_aligment_test['phone_rate'] = charsiu_pred_aligment_test['phone_rate'].where(charsiu_pred_aligment_test['utterance'] != '[SIL]', 0)


#%%
charsiu_df_by_sample_train = ut.TIMIT_df_by_sample_phones(charsiu_pred_aligment_train)

charsiu_df_by_sample_test = ut.TIMIT_df_by_sample_phones(charsiu_pred_aligment_test)

#%%
# Each column of charsiu_df_by_sample_train add the name charsiu_pred_aligment 
#charsiu_df_by_sample_train.columns = ['charsiu_pred_aligment_' + str(col) for col in charsiu_df_by_sample_train.columns]

# Merge with X_TRAIN
df_X_TRAIN['CHARSIU_mean_speed_wpau'] = charsiu_df_by_sample_train['mean_speed_wpau_v1']
df_X_TRAIN['CHARSIU_mean_speed_wopau'] = charsiu_df_by_sample_train['mean_speed_wopau_v1']

df_X_TEST['CHARSIU_mean_speed_wpau'] = charsiu_df_by_sample_test['mean_speed_wpau_v1']
df_X_TEST['CHARSIU_mean_speed_wopau'] = charsiu_df_by_sample_test['mean_speed_wopau_v1']


#%%


df_X = df_X_TRAIN



#%% -------------------------- SPLITTING -------------------------------------
speaker_id = df_X['speaker_id'].unique()
n_speakers = len(speaker_id)

# 80% train, 20% VAL

n_train = int(0.8*n_speakers)
n_val = int(0.2*n_speakers)

# Choose randomly set seed
random.seed(42)
random.shuffle(speaker_id)
speaker_id_train = speaker_id[:n_train]
speaker_id_val = speaker_id[n_train:n_train+n_val]


#%%
# Filter the speaker_id_train
df_TRAIN = df_X[df_X['speaker_id'].isin(speaker_id_train)]
df_VAL = df_X[df_X['speaker_id'].isin(speaker_id_val)]
df_TEST = df_X_TEST


# %% TRAIN SET
X_TRAIN  = df_TRAIN.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_TRAIN = df_TRAIN[['mean_speed_wpau_v1', 'mean_speed_wopau_v1']]
#%%

X_VAL = df_VAL.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_VAL = df_VAL[['mean_speed_wpau_v1', 'mean_speed_wopau_v1']]

X_TEST = df_X_TEST.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_TEST = df_X_TEST[['mean_speed_wpau_v1', 'mean_speed_wopau_v1']]


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

E = ['CHARSIU_mean_speed_wpau']


features = [A,B,B_abs,C,D,E,BC,B_ABS_C]

#%%
N=1

y_train = y_TRAIN['mean_speed_wpau_v1']
y_val = y_VAL['mean_speed_wpau_v1']
y_test = y_TEST['mean_speed_wpau_v1']
mean_score_features_wpau, mean_MSE_features_wpau, mode_wp, test_scores_wp= ut.r2_experiment(X_TRAIN, X_VAL,X_TEST, y_train, y_val, y_test, features, N=1)
#%%
y_train = y_TRAIN['mean_speed_wopau_v1']
y_val = y_VAL['mean_speed_wopau_v1']
y_test = y_TEST['mean_speed_wopau_v1']

mean_score_features_wopau, mean_MSE_features_wopau, model_wop, test_scores_wop = ut.r2_experiment(X_TRAIN, X_VAL,X_TEST, y_train, y_val, y_test, features, N=1)


#%%
# Do barplot of the features with MSE add one next to the other
x = np.arange(len(features))
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_score_features_wpau, width, label='With pauses v1')
rects2 = ax.bar(x + width/2, mean_score_features_wopau, width, label='With out pauses v1')
plt.xticks(np.arange(len(features)), ['A, dim:' + str(len(A)),'B, dim:'+ str(len(B)),'C, dim:'+ str(len(C)), 'D, dim:'+ str(len(D)), 'BC, dim:'+ str(len(BC)), 'F, dim:'+ str(len(E)), 'B_ABS_C, dim:'+ str(len(B_ABS_C))]
           , rotation=0)

# Add in this plot the name of each feature
plt.title('Prediction of the speech rate')
plt.xlabel('Groups of features')
plt.ylabel('Mean R2 - '+ str(N) + ' iterations')
plt.ylim(0, 1)
plt.legend()
plt.savefig('barplot_mean_speed.png')
plt.show()

# %% Barplot with more details

import matplotlib.pyplot as plt
import numpy as np

# Sample data
features = [A,B,B_abs,C,D,E,BC,B_ABS_C]

dimensions = [len(A), len(B),len(B_abs), len(C), len(D), len(E), len(BC), len(B_ABS_C)]

y_train = y_TRAIN['mean_speed_wpau_v1']
y_val = y_VAL['mean_speed_wpau_v1']
y_test = y_TEST['mean_speed_wpau_v1']


mean_score_features_wpau, mean_MSE_features_wpau, mode_wp, test_scores_wp= ut.r2_experiment(X_TRAIN, X_VAL,X_TEST, y_train, y_val, y_test, features, N=1)
#%%
y_train = y_TRAIN['mean_speed_wopau_v1']
y_val = y_VAL['mean_speed_wopau_v1']
y_test = y_TEST['mean_speed_wopau_v1']

mean_score_features_wopau, mean_MSE_features_wopau, model_wop, test_scores_wop = ut.r2_experiment(X_TRAIN, X_VAL,X_TEST, y_train, y_val, y_test, features, N=1)

##%%
## Model WP in test
#
#test_scores_wp = []
#test_scores_wop = []
#for i in range(len(features)):
#     model_wp.fit(X_TRAIN[features[i]], y_TRAIN['mean_speed_wpau_v1'])
#     test_scores_wp.append(model_wp.score(X_TEST[features[i]], y_TEST['mean_speed_wpau_v1']))
#
#     model_wop.fit(X_TRAIN[features[i]], y_TRAIN['mean_speed_wopau_v1'])
#     test_scores_wop.append(model_wop.score(X_TEST[features[i]], y_TEST['mean_speed_wopau_v1']))


#%%
x = np.arange(len(features))
width = 0.15  # Adjusted width to fit six sets within the plot

fig, ax = plt.subplots(figsize=(14, 8))  # Set figure size for better visualization
rects1 = ax.bar(x - 3*width/2, mean_score_features_wpau, width, label='Entrenamiento con Pausas', color='deepskyblue')
rects2 = ax.bar(x - width/2, mean_score_features_wopau, width, label='Entrenamiento sin Pausas', color='dodgerblue')
rects3 = ax.bar(x + width/2, test_scores_wp, width, label='Testeo con Pausas', color='sandybrown')
rects4 = ax.bar(x + 3*width/2, test_scores_wop, width, label='Testeo sin Pausas', color='darkorange')

# Improve x-ticks
plt.xticks(np.arange(len(features)), ['A, Dim:' + str(dimensions[0]), 'B, Dim:' + str(dimensions[1]), '|B|, Dim:' + str(dimensions[2]), 'C, Dim:' + str(dimensions[3]), 'D, Dim:' + str(dimensions[4]), 'E, Dim:' + str(dimensions[5]), 'BC, Dim:' + str(dimensions[6]), '|B|C, Dim:' + str(dimensions[7])], rotation=0) 
                                             

plt.title('Predicciones de la velocidad del habla', fontsize=20)
plt.xlabel('Grupos de Atributos', fontsize=24)
plt.ylabel('Coeficiente de Determinación (R2)', fontsize=20)
plt.ylim(0, 1)  # Adjust ylim to enhance visibility of labels
# increase the font size of the legend

plt.legend(fontsize='17', title_fontsize='100')

  # Relocate legend to avoid overlap with data

plt.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')  # Add grid lines for better measurement estimation
plt.tight_layout()  # Adjust layout to ensure no overlap of text/labels
plt.savefig('experiment_1.png')

plt.show()




# %% Correlation Matrix

# Compute the correlation matrix
corr = df_TRAIN[A+D+E].corr()

# plot the heatmap
sns.heatmap(corr, 
        xticklabels=corr.columns,
        yticklabels=corr.columns, cmap='coolwarm')
# %%
