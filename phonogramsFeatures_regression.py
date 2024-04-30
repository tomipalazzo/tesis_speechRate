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


#%% Load TIMIT
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Load the Dataframes

record_phone_test = pd.read_csv('TIMIT_df_by_record_phone_test.csv')
record_phone_train = pd.read_csv('TIMIT_df_by_record_phone_train.csv')
sample_phone_train = pd.read_csv('TIMIT_df_by_sample_train.csv')
sample_phone_test = pd.read_csv('TIMIT_df_by_sample_test.csv')

#%%
sample_phone_train
#%% Global Variables
N_SAMPLES = 10


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
def get_phonograms(TIMIT_set, model, n_samples = N_SAMPLES,  train=True):
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
    # save the list of phonograms
    #phonograms_df = pd.DataFrame()
    #phonograms_df['sample_id'] = [sample[0] for sample in phonograms]
    #phonograms_df['phonogram'] = [sample[1] for sample in phonograms]
    #phonograms_df['phonogram'] = [np.array2string(sample[1]) for sample in phonograms]
    #if train:
    #    phonograms_df.to_csv('../tesis_speechRate/data_phonograms/phonograms_CHARSIU_df_TRAIN_test.csv', index=False)
    #else:
    #    phonograms_df.to_csv('../tesis_speechRate/data_phonograms/phonograms_CHARSIU_df_TEST.csv', index=False)
    #return phonograms_df

# Dataframes with the array of the samples (array_id, array)

def get_sample_ID(dialect_region, speaker_id, id):
    return dialect_region + '_' + speaker_id + '_' + id

def get_sample_IDs(TIMIT_set, n_samples = N_SAMPLES):
    sample_id = []
    for i in range(n_samples):
        sample = TIMIT_set[i]
        sample_id.append(get_sample_ID(sample['dialect_region'], sample['speaker_id'], sample['id']))
    return sample_id


#  Phonogram features
def phonogram_to_features(phonogram, sample_id):
    delta = librosa.feature.delta(phonogram)
    d_delta = librosa.feature.delta(phonogram, order=2)

    for i in range(phonogram.shape[0]):
        dic = {'sample_id': sample_id}
        for j in range(1,42+1): # 42 is the number of phonemes
            dic['mean_phonogram_' + str(j)] = np.mean(phonogram[i,:])
            dic['mean_delta_' + str(j)] = np.mean(delta[i,:])
            dic['mean_d_delta_' + str(j)] = np.mean(d_delta[i,:])
            dic['std_phonogram_' + str(j)] = np.std(phonogram[i,:])
            dic['std_delta_' + str(j)] = np.std(delta[i,:])
            dic['std_d_delta_' + str(j)] = np.std(d_delta[i,:])
            dic['abs_mean_phonogram_' + str(j)] = np.mean(np.abs(phonogram[i,:]))
            dic['abs_mean_delta_' + str(j)] = np.mean(np.abs(delta[i,:]))
            dic['abs_mean_d_delta_' + str(j)] = np.mean(np.abs(d_delta[i,:]))
            dic['mean_all_phonogram'] = np.mean(phonogram)

    features = pd.DataFrame(dic, index=[0])
    return features 

def feature_phonograms(phonogram):
    phonogram_softmaxed = torch.softmax(torch.tensor(phonogram), dim=1).detach().numpy()   

#%% ------------------------------------------------------------------

# Get phonogram features of N_SAMPLES samples in the training set
data_phonograms_TRAIN = pd.read_csv('../tesis_speechRate/data_phonograms/phonograms_features_df_TRAIN.csv')
data_phonograms_TEST = pd.read_csv('../tesis_speechRate/data_phonograms/phonograms_features_df_TEST.csv')
data_phonograms_Charsiu_TRAIN = pd.read_csv('../tesis_speechRate/data_phonograms/phonograms_CHARSIU_df_TRAIN.csv')
data_phonograms_Charsiu_TEST = pd.read_csv('../tesis_speechRate/data_phonograms/phonograms_CHARSIU_df_TEST.csv')
#%% Now with Test
data_phonograms_TRAIN.set_index('sample_id', inplace=True)
data_phonograms_TEST.set_index('sample_id', inplace=True)
#phonograms_features_df
#%% Parcing Matrix
# Apply the conversion function to the DataFrame column


data_phonograms_Charsiu_TRAIN['phonogram'] = data_phonograms_Charsiu_TRAIN['phonogram'].apply(parse_matrix_from_string) 


#%%

sample_phone_train.set_index('sample_id', inplace=True)

# %%
# Get phonograms TRAIN
TRAIN_ROWS = len(TIMIT_train)
test = get_phonograms(TIMIT_train, modelo, TRAIN_ROWS, train=True)

#%% Get phonograms TEST
TEST_ROWS = len(TIMIT_test)
get_phonograms(TIMIT_test, modelo, TEST_ROWS, train=False)
# %%
df_X = pd.merge(data_phonograms_TRAIN, sample_phone_train, left_index=True, right_index=True)
# %%
X = df_X.drop(columns=['mean_speed_wpau', 'mean_speed_wopau', 'duration_wpau'])
y = df_X['mean_speed_wpau']

#%% 
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
print(model.score(X_test_pca, y_test))
print(model.score(X_train_pca, y_train)
)
plt.figure()
plt.scatter(X_train_pca, y_train,alpha=0.2)
plt.title('Linear Regression with PCA (1 component)')
plt.xlabel('PCA component')
plt.ylabel('Speed')
plt.show()
# %%
df_X.corr()
# %%
import seaborn as sns
sns.heatmap(df_X.corr())
# %%
sample_phone_test.set_index('sample_id', inplace=True)
#%%
df_X_test = pd.merge(phonograms_features_df_test, sample_phone_test, left_index=True, right_index=True)
sns.heatmap(df_X_test.corr())


# %% 
plt.scatter(X_test_pca, y_test, alpha=0.2)
# %%
model = linear_model.LinearRegression()
model.fit(X_train_pca, y_train)
y_pred_train = model.predict(X_train_pca)
y_pred_test = model.predict(X_test_pca)  
print(mean_squared_error(y_train, y_pred_train))
print(mean_squared_error(y_test, y_pred))

# %%
print(model.score(X_train_pca, y_train))
print(model.score(X_test_pca, y_test))
# %%
