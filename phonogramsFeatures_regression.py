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
import ast  # For safely evaluating strings containing Python literals

# ---------------------------- LOAD DATASET -----------------------------------
#%% Load TIMIT
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Load the Dataframes

record_phone_test = pd.read_csv('TIMIT_df_by_record_phone_test.csv')
record_phone_train = pd.read_csv('TIMIT_df_by_record_phone_train.csv')
sample_phone_train = pd.read_csv('TIMIT_df_by_sample_train.csv')
sample_phone_test = pd.read_csv('TIMIT_df_by_sample_test.csv')


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
# Example: Forced alignment of an audio file
x = torch.tensor(np.array([TIMIT_train[0]['audio']['array']]).astype(np.float32))
with torch.no_grad():
    y = modelo(x).logits
y = y.numpy()[0].T
plt.pcolor(y)
# ----------------------------------------------------------------------------



#%% --------------------- FUNCTIONS ------------------------------------------
def get_phonograms(TIMIT_set, model, n_samples = 10,  train=True):
    t0 = time.time()
    phonograms = []
    

    print('--------------GETTING PHONOGRAMS-----------------')
    for i in range(n_samples):
        sample = TIMIT_set[i]

        sample_id = get_sample_ID(sample['dialect_region'], sample['speaker_id'], sample['id'])
        
        audio_data = sample['audio']['array']
        x = torch.tensor(np.array([audio_data]).astype(np.float32))
        with torch.no_grad():
            y = model(x).logits
        y = y.numpy()[0].T

        # Save each phonogram as a matrix the ohonograms are numpy.ndarray
        y_df = pd.DataFrame(y)
        if train:
            y_df.to_csv('../tesis_speechRate/data_phonograms/CHARSIU/Train/' + sample_id + '.csv', index=False)
        else:
            y_df.to_csv('../tesis_speechRate/data_phonograms/CHARSIU/Test/' + sample_id + '.csv', index=False)
        
        
       # phonograms.append([sample_id,y]) 
        if i % 10 == 0:

            print('SAMPLE ', i, ' OF ', n_samples)

    t1 = time.time()
    print('-------------------------------------------------')
    print('Time to get phonograms: ', t1-t0)

# Dataframes with the array of the samples (array_id, array)

def get_sample_ID(dialect_region, speaker_id, id):
    return dialect_region + '_' + speaker_id + '_' + id

def get_sample_IDs(TIMIT_set, n_samples = 10):
    sample_id = []
    for i in range(n_samples):
        sample = TIMIT_set[i]
        sample_id.append(get_sample_ID(sample['dialect_region'], sample['speaker_id'], sample['id']))
    return sample_id


#  Phonogram features
def phonogram_to_features(sample_ID, train=True):
    if train:
        phonogram = pd.read_csv('../tesis_speechRate/data_phonograms/CHARSIU/Train/'+sample_ID+'.csv')
    else: 
        phonogram = pd.read_csv('../tesis_speechRate/data_phonograms/CHARSIU/Test/'+sample_ID+'.csv')
    
    phonogram = phonogram.to_numpy()
    delta = librosa.feature.delta(phonogram)
    d_delta = librosa.feature.delta(phonogram, order=2)


    dic = {'sample_id': sample_ID}
    for j in range(42): # 42 is the number of phonemes
        dic['mean_phone_' + str(j+1)] = np.mean(phonogram[j,:])
        dic['mean_delta_' + str(j+1)] = np.mean(delta[j,:])
        dic['mean_d_delta_' + str(j+1)] = np.mean(d_delta[j,:])
        dic['std_phone_' + str(j+1)] = np.std(phonogram[j,:])
        dic['std_delta_' + str(j+1)] = np.std(delta[j,:])
        dic['std_d_delta_' + str(j+1)] = np.std(d_delta[j,:])
        dic['abs_mean_phone_' + str(j+1)] = np.mean(np.abs(phonogram[j,:]))
        dic['abs_mean_delta_' + str(j+1)] = np.mean(np.abs(delta[j,:]))
        dic['abs_mean_d_delta_' + str(j+1)] = np.mean(np.abs(d_delta[j,:]))
    dic['mean_all_phonogram'] = np.mean(phonogram)
    dic['mean_all_delta'] = np.mean(delta)
    dic['mean_all_d_delta'] = np.mean(d_delta)
    dic['std_all_phonogram'] = np.std(phonogram)
    dic['abs_all_phonogram'] = np.mean(np.abs(phonogram))
    dic['abs_all_delta'] = np.mean(np.abs(delta))
    dic['abs_all_d_delta'] = np.mean(np.abs(d_delta))

    features = pd.DataFrame(dic, index=[0])
    return features

def phonograms_to_features(sample_IDs, train = True):
    print('==============GETTING PHONOGRAMS FEATURES================')
    features = pd.DataFrame()
    for i in range(len(sample_IDs)):
        features = pd.concat([features, phonogram_to_features(sample_IDs[i], train=train)], ignore_index=True)

        if i % 10 == 0:
            print('SAMPLE ', i, ' OF ', len(sample_IDs))
            print('-------------------------------------------------')

    print('=================FINISHED===================')        
    return features

def feature_softmax(phonogram):
    phonogram_softmaxed = torch.softmax(torch.tensor(phonogram), dim=1).detach().numpy()   


#%% -------------------------- GENERATE PHONOGRAMS -----------------------------
# OBS: This process takes a long time. It save the phonograms as csv files in the data_phonograms folder 

# get_phonograms(TIMIT_train, modelo, N_TRAIN, train=True)



# =======Get phonograms TEST============= UNCOMMENT TO GET PHONOGRAMS

#get_phonograms(TIMIT_test, modelo, N_TEST, train=False)


#%% ---------------------------- GLOBAL VARIABLES --------------------------------

N_TRAIN = len(TIMIT_train)
N_TEST = len(TIMIT_test)
SAMPLE_IDs_TRAIN = get_sample_IDs(TIMIT_train, N_TRAIN)
SAMPLE_IDs_TEST = get_sample_IDs(TIMIT_test, N_TEST)




#%% -------------------------- GENERATE FEATURES -----------------------------

# Get phonogram features of N_SAMPLES samples in the training set

phonogram_features_TRAIN = phonograms_to_features(SAMPLE_IDs_TRAIN, train = True)
phonogram_features_TEST = phonograms_to_features(SAMPLE_IDs_TEST, train = False)
#%%
phonogram_features_TRAIN.set_index('sample_id', inplace=True)
phonogram_features_TEST.set_index('sample_id', inplace=True)

#%% -------------------------- REGRESSION -------------------------------------

# Merge the phonogram features with the sample_phone dataframes
sample_phone_train.set_index('sample_id', inplace=True)
sample_phone_test.set_index('sample_id', inplace=True)
#%%
df_X_TRAIN = pd.merge(phonogram_features_TRAIN, sample_phone_train, left_index=True, right_index=True)
# %% TRAIN SET
X_TRAIN = df_X_TRAIN.drop(columns=['mean_speed_wpau', 'mean_speed_wopau', 'duration_wpau'])
y_TRAIN = df_X_TRAIN['mean_speed_wpau']


#%% TEST SET
df_X_TEST = pd.merge(phonogram_features_TEST, sample_phone_test, left_index=True, right_index=True)
X_TEST = df_X_TEST.drop(columns=['mean_speed_wpau', 'mean_speed_wopau', 'duration_wpau'])
y_TEST = df_X_TEST['mean_speed_wpau']
#%% REGRESSION in TRAINING SET
X_train, X_test, y_train, y_test = train_test_split(X_TRAIN, y_TRAIN, test_size=0.2)
positive=True
model = linear_model.LinearRegression(positive=positive)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mean_squared_error(y_test, y_pred)

#%%
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

print('========= REGRESSION ALL FEATURES ' + 'POSITIVE=' + str(positive) + '=========')
print('MSE(y_train, y_train_pred):',mean_squared_error(y_train, y_train_pred))
print('MSE(y_test, y_test_pred):',mean_squared_error(y_test, y_test_pred))
# R2 score
print('SCORE(X_train, y_train):',model.score(X_train, y_train))
print('SCORE(X_test, y_test):',model.score(X_test, y_test))




#%% ======================== PCA =============================================== 
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(X_train)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.grid()
# 95% of the variance
dimention = np.where(np.cumsum(pca.explained_variance_ratio_) < 0.95)

#%%
pca = PCA(n_components=40)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
positive=False
model = linear_model.LinearRegression(positive=positive)
model.fit(X_train_pca, y_train)
y_pred = model.predict(X_test_pca)
print('=========PCA 40 components ' + 'POSITIVE=' + str(positive) + '=========')
print('MSE(y_test, y_pred):',mean_squared_error(y_test, y_pred))
print('MSE(y_train, y_train_pred):',mean_squared_error(y_train, model.predict(X_train_pca)))
print('SCORE(X_test, y_test):',model.score(X_test_pca, y_test))
print('SCORE(X_train, y_train):',model.score(X_train_pca, y_train))

# %%
df_X_TRAIN.corr()
# %%
import seaborn as sns
sns.heatmap(df_X_TRAIN.corr())
# %%
sample_phone_test.set_index('sample_id', inplace=True)
#%%
df_X_test = pd.merge(phonograms_features_df_test, sample_phone_test, left_index=True, right_index=True)
sns.heatmap(df_X_test.corr())

# %% =================== FEATURES SELECTION ===================================

from sklearn.feature_selection import SelectKBest

# Select the K best features
K = 20
selector = SelectKBest(k=K)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
model = linear_model.LinearRegression(positive=True)
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

selected_features_mask = selector.get_support()
selected_features = X_train.columns[selected_features_mask]

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

positive = False
model = linear_model.LinearRegression(positive=positive)
model.fit(X_train_selected, y_train)
y_pred = model.predict(X_test_selected)

print('=========SELECTED FEATURES ' + 'K=' + str(K) + ' POSITIVE=' + str(positive) + '=========')
print('MSE(y_test, y_pred):', mean_squared_error(y_test, y_pred))
print('MSE(y_train, y_train_pred):', mean_squared_error(y_train, model.predict(X_train_selected)))
print('SCORE(X_test, y_test):', model.score(X_test_selected, y_test))
print('SCORE(X_train, y_train):', model.score(X_train_selected, y_train))

# Correlation of the selected features
#sns.heatmap(X_train_selected.corr())
 
# %% =============== FEATURE SELECTION ========================================= 
ks = np.arange(1,len(df_X_TRAIN.columns)+1, 10)

MSE_k = []
for k in range(1,len(df_X_TRAIN.columns)+1, 10):
    print('K=', k, ' of ', len(df_X_TRAIN.columns))
    selector = SelectKBest(k=round(k))
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    model = linear_model.LinearRegression(positive=True)
    model.fit(X_train_selected, y_train)
    y_pred = model.predict(X_test_selected)

    MSE_k.append(mean_squared_error(y_test, y_pred))

#%%
# Do barplot of the features with MSE
plt.bar(ks, MSE_k)
# The line is thick
plt.plot(ks, MSE_k, color='red', linewidth=2)
plt.xticks(rotation=90)
# Number of k in x axis
plt.xlabel('Number of features')
# MSE in y axis
plt.ylabel('Mean Squared Error')
plt.show()



# %% =================== GROUPS OF FEATURES HAND PICKED ========================
# Hand picked features
features_00 = ['mean_all_phonogram']
features_01 = ['mean_all_phonogram','mean_all_delta', 'mean_all_d_delta', 'std_all_phonogram']
features_02 = ['mean_all_phonogram','mean_all_delta', 'std_all_phonogram']
features_03 = ['mean_all_phonogram', 'mean_all_delta']

# Try each feature
MSE_features = []
for i in range(4):
    if i == 0:
        features_i = features_00
    elif i == 1:
        features_i = features_01
    elif i == 2:
        features_i = features_02
    else:
        features_i = features_03
    print('Features:', features_i)
    


    
        


    y_TRAIN = df_X_TRAIN['mean_speed_wpau']
    df_X_TRAIN_fi = df_X_TRAIN[features_i]
    
    # Regression
    X_train, X_test, y_train, y_test = train_test_split(X_TRAIN, y_TRAIN, test_size=0.2)
    positive=True
    model = linear_model.LinearRegression(positive=positive)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    MSE_features.append(mean_squared_error(y_test, y_pred))

# Do barplot of the features with MSE
plt.bar(np.arange(4), MSE_features)

# Matrix of correlation


# %%
