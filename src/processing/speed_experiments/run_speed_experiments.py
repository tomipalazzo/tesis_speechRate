#%%
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
from src.pre_processing.metrics.metrics import m1


# %%
# Idea: Prediction of the speech rate
# 1. Ground truth: m1 from pre-processing/metrics/run_metrics.py
# 2. Model: TODO model that predicts the speech rate
# 3. Data: a. Phonograms from CHARSIU
#          b. Sections of those phonograms
#          c. Features of this sections
#4. Evaluation: a. MSE
#               b. R2
#               c. Visual comparison of the predicted speech rate and the ground truth

#%% Load the data
# Load the data
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Load the Dataframes in the correct directory

record_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_test.csv')
record_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_train.csv')
sample_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_train.csv')
sample_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_test.csv')

#%% ======================== FUNCTIONS ========================

def get_real_speed(sample_ids, train=True, step_size=1/60*16000, window_size=1/4*16000, with_pause=False):
    """
    This function returns the real speed of the samples in the sample_ids list.
    """
    print('PROCESSING ' + 'TRAIN' if train else 'TEST')
    N = len(sample_ids)
    
    t0 = time.time()

    for i in range(N):

        if i % 100 == 0:
            print('Processing sample ' + str(i) + ' of ' + str(N))
        
        speed_df = pd.DataFrame()
        sample_id = sample_ids[i]
        sample_features = record_phone_train[record_phone_train['sample_id'] == sample_id] if train else record_phone_test[record_phone_test['sample_id'] == sample_id]
        #reset index
        sample_features = sample_features.reset_index(drop=True)
        
        x, y = m1(sample_features, step_size=step_size, window_size=window_size, with_pau=with_pause)
        speed_df['x'] = x
        speed_df['y'] = y

        # save as csv
        if train:
            speed_df.to_csv(f'../tesis_speechRate/src/processing/speed_experiments/data/ground_truth/train/{sample_id}.csv', index=False)
        else:
            speed_df.to_csv(f'../tesis_speechRate/src/processing/speed_experiments/data/ground_truth/test/{sample_id}.csv', index=False)
    
    tf = time.time()

    print('DONE. Time:' + str(tf-t0) + 's')

#%% GLOBAL VARIABLES

N_TRAIN = len(TIMIT_train)
N_TEST = len(TIMIT_test)
SAMPLE_IDs_TRAIN = ut.get_sample_IDs(TIMIT_train, N_TRAIN)
SAMPLE_IDs_TEST = ut.get_sample_IDs(TIMIT_test, N_TEST)


# %% ======================== GRUOND TRUTH ========================
N = 200
N = 200
step_size = 1/8*16000     #seg
window_size = 1/4*16000   #seg
with_pause = False


# Get the real speed of the samples
get_real_speed(SAMPLE_IDs_TRAIN[:N], train=True, step_size=step_size, window_size=window_size, with_pause=with_pause)


# %% read the csv
sample_id = SAMPLE_IDs_TRAIN[10]
speed_df = pd.read_csv(f'../tesis_speechRate/src/processing/speed_experiments/data/ground_truth/train/{sample_id}.csv')
x = speed_df['x']
y = speed_df['y']
plt.plot(x, y)
plt.xlabel('Time (s)')
plt.ylabel('Speed')
plt.title('Step size = ' + str(step_size/16000) + 's, Window size = ' + str(window_size/16000) + 's, with_pause = ' + str(with_pause))
# add labels to the plot
# Add the name of the element in the legen in format sample id = sample_id, step size = step_size, window_size = window_size, with_pause = with_pause


# %% ======================== DATA =======================
SAMPLE_id = SAMPLE_IDs_TRAIN[0]
phonogram = ut.get_phonograms_from_csv(sample_ID=SAMPLE_id, train=True)

plt.pcolor(phonogram)

#%%
def phonogram_window(phongram, window_size=20, step_size=5):
    """
    This function a lot of sections of the phonogram.
    """
    N = phongram.shape[1]
    sections = []
    times = np.arange(0, N, step_size)

    for i in range(len(times)):
        left = int(max(0, times[i] - window_size/2))
        right = int(min(N-1, times[i] + window_size/2))
        section = phongram[:, left:right]
        sections.append(section)
    
    return sections
# %% Test function
sections = phonogram_window(phonogram)
len(sections)
# %%

plt.pcolor(sections[0])

# %%


# %%
def phonogram_to_features(sample_ID,phonogram, train=True, begin=0, end=0):


    phonogram = phonogram
    if end == 0:
        end = phonogram.shape[1]
    phonogram = phonogram[:, begin:end]
    delta = librosa.feature.delta(phonogram)
    d_delta = librosa.feature.delta(phonogram, order=2)
    
    DR_ID, speaker_ID = ut.get_dialectRegion_and_speacker_ID(sample_ID=sample_ID)

    dic = {'sample_id': sample_ID}
    dic['region_id'] =  DR_ID
    dic['speaker_id'] = speaker_ID
    
    # TODO: Vectorizar al estilo numpy

    # Mean each phonogram
    means = np.mean(phonogram, axis=1)
    mean_dic = {f'mean_phone_{i+1}': mean for i, mean in enumerate(means)}
    dic.update(mean_dic)

    # STD each phonogram
    stds = np.std(phonogram, axis=1)
    std_dic = {f'std_phone_{i+1}': std for i, std in enumerate(stds)}
    dic.update(std_dic)
    
    # STD delta each phonogram
    mean_delta = np.mean(delta, axis=1)
    mean_delta_dic = {f'mean_delta_phone_{i+1}': mean for i, mean in enumerate(mean_delta)}
    dic.update(mean_delta_dic)

    # STD delta each phonogram
    std_delta = np.std(delta, axis=1)
    std_delta_dic = {f'std_delta_phone_{i+1}': std for i, std in enumerate(std_delta)}
    dic.update(std_delta_dic)

    # STD delta each phonogram
    mean_d_delta = np.mean(d_delta, axis=1)
    mean_d_delta_dic = {f'mean_d_delta_phone_{i+1}': mean for i, mean in enumerate(mean_d_delta)}
    dic.update(mean_d_delta_dic)

    # STD delta each phonogram
    std_d_delta = np.std(d_delta, axis=1)
    std_d_delta_dic = {f'std_d_delta_phone_{i+1}': std for i, std in enumerate(std_d_delta)}
    dic.update(std_d_delta_dic)

    # Mean absolute value of each phonogram
    abs_mean_phonogram = np.mean(np.abs(phonogram), axis=1)
    abs_mean_phonogram_dic = {f'mean_abs_phone_{i+1}': mean for i, mean in enumerate(abs_mean_phonogram)}
    dic.update(abs_mean_phonogram_dic)

    # Mean absolute value of each delta
    mean_abs_delta = np.mean(np.abs(delta), axis=1)
    mean_abs_delta_dic = {f'mean_abs_delta_phone_{i+1}': mean for i, mean in enumerate(mean_abs_delta)}
    dic.update(mean_abs_delta_dic)

    # Mean absolute value of each delta
    mean_abs_d_delta = np.mean(np.abs(d_delta), axis=1)
    mean_abs_d_delta_dic = {f'mean_abs_d_delta_phone_{i+1}': mean for i, mean in enumerate(mean_abs_d_delta)}
    dic.update(mean_abs_d_delta_dic)

    # Add the feature of softmax
    feature_softmax = ut.how_many_probables_phones(phonogram)[0]
    dic_feature_softmax = {f'feature_softmax_{i+1}': feature for i, feature in enumerate(feature_softmax)}
    dic.update(dic_feature_softmax)

    # Add the mean feature of softmax
    mean_feature_softmax = ut.how_many_probables_phones(phonogram)[1]
    dic_mean_feature_softmax = {f'mean_feature_softmax_{i+1}': feature for i, feature in enumerate(mean_feature_softmax)}
    dic.update(dic_mean_feature_softmax)    


    # Means realated of all the phonogram
    dic['all_mean_phonogram'] = np.mean(phonogram)
    dic['all_mean_delta'] = np.mean(delta)
    dic['all_mean_d_delta'] = np.mean(d_delta)
    dic['all_std_phonogram'] = np.std(phonogram)
    dic['all_mean_abs_phonogram'] = np.mean(np.abs(phonogram))
    dic['all_mean_abs_delta'] = np.mean(np.abs(delta))
    dic['all_mean_abs_d_delta'] = np.mean(np.abs(d_delta))
    
    dic['greedy_feature'] = ut.greedy_feature(phonogram)[0]
    
    
    features = pd.DataFrame(dic, index=[0])
    # Save the features as a csv file
    return features

# %% =========================== CREATE THE DATA SET =================================================
def speed_df(sample_id, train=True, step_size=5, window_size=20, with_pause=False):
    """
    This function creates the dataset for the sample_id.
    """

    # Get the phonogram
    phonogram = ut.get_phonograms_from_csv(sample_ID=sample_id, train=train)
    phonogram_lenght = phonogram.shape[1] # Sample rate 10ms

    step_phonogram = step_size
    window_phonogram = window_size
    sections = phonogram_window(phonogram, window_size=window_phonogram, step_size=step_phonogram)


    features_df = pd.DataFrame()

    for i in range(len(sections)):
        sectioni = sections[i]
        features = phonogram_to_features(sample_ID=sample_id,phonogram=sectioni, train=True)
        features_df = pd.concat([features_df, features], ignore_index=True)
    

    
    # Get the real speed
    step_y  = step_phonogram/100*16000    # In the same scale as the phonogram
    window_y = window_phonogram/100*16000 # In the same scale as the phonogram


    sample_features = record_phone_train[record_phone_train['sample_id'] == sample_id] if train else record_phone_test[record_phone_test['sample_id'] == sample_id]
    #reset index
    sample_features = sample_features.reset_index(drop=True)
    x, y = m1(sample_features, step_size=step_y, window_size=window_y, with_pau=with_pause)    


    df = pd.DataFrame()
    if len(x) == len(y) == len(sections):
        df['x'] = x
        df['y'] = y
        df['section'] = sections
        
    else:
        L  = min(len(x), len(sections), len(y))
        df['x'] = x[:L]
        df['y'] = y[:L]
        df['section'] = sections[:L]
    

    df = pd.merge(df, features_df, left_index=True, right_index=True)

    return df
#%%
def generate_data(sample_ids, train=True, step_size=5, window_size=20, with_pause=False):
    """
    This function generates the dataset for the sample_ids.
    """
    N = len(sample_ids)
    t0 = time.time()
    data_set_with_out_pau = pd.DataFrame()
    for i in range(N):
        if i % 10 == 0:
            print('Processing sample ' + str(i) + ' of ' + str(N))
        
        sample_id = sample_ids[i]
        df = speed_df(sample_id, train=train, step_size=step_size, window_size=window_size, with_pause=with_pause)
        data_set_with_out_pau = pd.concat([data_set_with_out_pau, df], ignore_index=True)

    if train:
        if with_pause:
            data_set_with_out_pau.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_train_with_pau.csv', index=False)
        else:
            data_set_with_out_pau.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_train_with_out_pau.csv', index=False)
    else:
        if with_pause:
            data_set_with_out_pau.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_test_with_pau.csv', index=False)
        else:
            data_set_with_out_pau.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_test_with_out_pau.csv', index=False)
    tf = time.time()
    print('DONE. Time:' + str(tf-t0) + 's')




# %%
SAMPLE_ID = SAMPLE_IDs_TRAIN[3]
# %%
# %%
#generate_data(SAMPLE_IDs_TRAIN, train=True, step_size=1/8*16000, window_size=1/4*16000, with_pause=False)
#generate_data(SAMPLE_IDs_TRAIN, train=True, with_pause=True)


# %%

# %% 



# %%
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


# %%
features = [A,B, C, D]



# Split
N=3


def get_by_feature(X, y, features, N=3):
    MSE_features = np.zeros(len(features))
    scores = np.zeros(len(features))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for j in range(N):
        for i in range(len(features)):
            print('Features:', features[i])
            X_train_fi = X_train[features[i]]
            X_test_fi = X_test[features[i]]
            
            # Regression
            positive=True
            model = linear_model.LinearRegression(positive=positive)
            model.fit(X_train_fi, y_train)
            y_pred = model.predict(X_test_fi)
            MSE_features[i] += mean_squared_error(y_test, y_pred)
            scores[i] += model.score(X_test_fi, y_test)
    
    mean_score_features = scores/N
    mean_MSE_features = MSE_features/N

    return mean_score_features, mean_MSE_features

mean_score_features_wpau, mean_MSE_features_wpau = get_by_feature(X_wpau, y_wpau, features, N=3)
mean_score_features_wo_pau, mean_MSE_features_wo_pau = get_by_feature(X_wopau, y_wopau, features, N=3)

# %%
x = np.arange(len(features))
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_score_features_wpau, width, label='With pauses v1')
plt.xticks(np.arange(len(features)), ['A, dim:' + str(len(A)),'B, dim:'+ str(len(B)),'C, dim:'+ str(len(C)), 'D, dim:'+ str(len(D))]
           , rotation=0)

# Add in this plot the name of each feature
plt.title('Prediction of speed for any time')
plt.xlabel('Groups of features')
plt.ylabel('Mean R2 - '+ str(N) + ' iterations')
plt.legend()
plt.savefig('a_b_c_d_barplot.png')
plt.show()
plt.pcolor(sections[0])

# %%


# %%
def phonogram_to_features(sample_ID,phonogram, train=True, begin=0, end=0):


    phonogram = phonogram
    if end == 0:
        end = phonogram.shape[1]
    phonogram = phonogram[:, begin:end]
    delta = librosa.feature.delta(phonogram)
    d_delta = librosa.feature.delta(phonogram, order=2)
    
    DR_ID, speaker_ID = ut.get_dialectRegion_and_speacker_ID(sample_ID=sample_ID)

    dic = {'sample_id': sample_ID}
    dic['region_id'] =  DR_ID
    dic['speaker_id'] = speaker_ID
    
    # TODO: Vectorizar al estilo numpy

    # Mean each phonogram
    means = np.mean(phonogram, axis=1)
    mean_dic = {f'mean_phone_{i+1}': mean for i, mean in enumerate(means)}
    dic.update(mean_dic)

    # STD each phonogram
    stds = np.std(phonogram, axis=1)
    std_dic = {f'std_phone_{i+1}': std for i, std in enumerate(stds)}
    dic.update(std_dic)
    
    # STD delta each phonogram
    mean_delta = np.mean(delta, axis=1)
    mean_delta_dic = {f'mean_delta_phone_{i+1}': mean for i, mean in enumerate(mean_delta)}
    dic.update(mean_delta_dic)

    # STD delta each phonogram
    std_delta = np.std(delta, axis=1)
    std_delta_dic = {f'std_delta_phone_{i+1}': std for i, std in enumerate(std_delta)}
    dic.update(std_delta_dic)

    # STD delta each phonogram
    mean_d_delta = np.mean(d_delta, axis=1)
    mean_d_delta_dic = {f'mean_d_delta_phone_{i+1}': mean for i, mean in enumerate(mean_d_delta)}
    dic.update(mean_d_delta_dic)

    # STD delta each phonogram
    std_d_delta = np.std(d_delta, axis=1)
    std_d_delta_dic = {f'std_d_delta_phone_{i+1}': std for i, std in enumerate(std_d_delta)}
    dic.update(std_d_delta_dic)

    # Mean absolute value of each phonogram
    abs_mean_phonogram = np.mean(np.abs(phonogram), axis=1)
    abs_mean_phonogram_dic = {f'mean_abs_phone_{i+1}': mean for i, mean in enumerate(abs_mean_phonogram)}
    dic.update(abs_mean_phonogram_dic)

    # Mean absolute value of each delta
    mean_abs_delta = np.mean(np.abs(delta), axis=1)
    mean_abs_delta_dic = {f'mean_abs_delta_phone_{i+1}': mean for i, mean in enumerate(mean_abs_delta)}
    dic.update(mean_abs_delta_dic)

    # Mean absolute value of each delta
    mean_abs_d_delta = np.mean(np.abs(d_delta), axis=1)
    mean_abs_d_delta_dic = {f'mean_abs_d_delta_phone_{i+1}': mean for i, mean in enumerate(mean_abs_d_delta)}
    dic.update(mean_abs_d_delta_dic)

    # Add the feature of softmax
    feature_softmax = ut.how_many_probables_phones(phonogram)[0]
    dic_feature_softmax = {f'feature_softmax_{i+1}': feature for i, feature in enumerate(feature_softmax)}
    dic.update(dic_feature_softmax)

    # Add the mean feature of softmax
    mean_feature_softmax = ut.how_many_probables_phones(phonogram)[1]
    dic_mean_feature_softmax = {f'mean_feature_softmax_{i+1}': feature for i, feature in enumerate(mean_feature_softmax)}
    dic.update(dic_mean_feature_softmax)    


    # Means realated of all the phonogram
    dic['all_mean_phonogram'] = np.mean(phonogram)
    dic['all_mean_delta'] = np.mean(delta)
    dic['all_mean_d_delta'] = np.mean(d_delta)
    dic['all_std_phonogram'] = np.std(phonogram)
    dic['all_mean_abs_phonogram'] = np.mean(np.abs(phonogram))
    dic['all_mean_abs_delta'] = np.mean(np.abs(delta))
    dic['all_mean_abs_d_delta'] = np.mean(np.abs(d_delta))
    
    dic['greedy_feature'] = ut.greedy_feature(phonogram)[0]
    
    
    features = pd.DataFrame(dic, index=[0])
    # Save the features as a csv file
    return features

# %% =========================== CREATE THE DATA SET =================================================
def speed_df(sample_id, train=True, step_size=5, window_size=20, with_pause=False):
    """
    This function creates the dataset for the sample_id.
    """

    # Get the phonogram
    phonogram = ut.get_phonograms_from_csv(sample_ID=sample_id, train=train)
    phonogram_lenght = phonogram.shape[1] # Sample rate 10ms

    step_phonogram = step_size
    window_phonogram = window_size
    sections = phonogram_window(phonogram, window_size=window_phonogram, step_size=step_phonogram)


    features_df = pd.DataFrame()

    for i in range(len(sections)):
        sectioni = sections[i]
        features = phonogram_to_features(sample_ID=sample_id,phonogram=sectioni, train=True)
        features_df = pd.concat([features_df, features], ignore_index=True)
    

    
    # Get the real speed
    step_y  = step_phonogram/100*16000    # In the same scale as the phonogram
    window_y = window_phonogram/100*16000 # In the same scale as the phonogram


    sample_features = record_phone_train[record_phone_train['sample_id'] == sample_id] if train else record_phone_test[record_phone_test['sample_id'] == sample_id]
    #reset index
    sample_features = sample_features.reset_index(drop=True)
    x_wp, y_wp = m1(sample_features, step_size=step_y, window_size=window_y, with_pau=True)    
    x_wop, y_wop = m1(sample_features, step_size=step_y, window_size=window_y, with_pau=False)

    df = pd.DataFrame()
    if len(x_wp) == len(y_wp) == len(x_wop) == len(y_wop) == len(sections):
        df['x_wp'] = x_wp
        df['y_wp'] = y_wp
        df['x_wop'] = x_wop
        df['y_wop'] = y_wop
        df['section'] = sections
        
    else:
        L  = min(len(x_wp), len(sections), len(y_wp), len(x_wop), len(x_wop))
        df['x_wp'] = x_wp[:L]
        df['y_wp'] = y_wp[:L]
        df['x_wop'] = x_wop[:L]
        df['y_wop'] = y_wop[:L]
        
        df['section'] = sections[:L]
    

    df = pd.merge(df, features_df, left_index=True, right_index=True)

    return df
#%%
def generate_data(sample_ids, train=True, step_size=5, window_size=20, with_pause=False):
    """
    This function generates the dataset for the sample_ids.
    """
    N = len(sample_ids)
    t0 = time.time()
    data_set = pd.DataFrame()
    for i in range(N):
        if i % 10 == 0:
            print('Processing sample ' + str(i) + ' of ' + str(N))
        
        sample_id = sample_ids[i]
        df = speed_df(sample_id, train=train, step_size=step_size, window_size=window_size, with_pause=with_pause)
        data_set = pd.concat([data_set, df], ignore_index=True)

    if train:
        if with_pause:
            data_set.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_train_with_pau.csv', index=False)
        else:
            data_set.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_train_with_out_pau.csv', index=False)
    else:
        if with_pause:
            data_set.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_test_with_pau.csv', index=False)
        else:
            data_set.to_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_test_with_out_pau.csv', index=False)
    tf = time.time()
    print('DONE. Time:' + str(tf-t0) + 's')




# %%
SAMPLE_ID = SAMPLE_IDs_TRAIN[3]
# %%
# %%
#generate_data(SAMPLE_IDs_TRAIN, train=True, with_pause=True)

# %% Open the data set
#data_set_with_out_pau = pd.read_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_train_with_out_pau.csv')
data_set_with_pau = pd.read_csv('../tesis_speechRate/src/processing/speed_experiments/data/data_set_train_with_pau.csv')
#
# %%
data_set_with_pau.shape




#%% -------------------------- SPLITTING -------------------------------------
speaker_id = data_set_with_pau['speaker_id'].unique()
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
df_TRAIN = data_set_with_pau[data_set_with_pau['speaker_id'].isin(speaker_id_train)]
df_VAL = data_set_with_pau[data_set_with_pau['speaker_id'].isin(speaker_id_val)]


# %% TRAIN SET
X_TRAIN  = df_TRAIN.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_TRAIN = df_TRAIN['mean_speed_wpau_v1']
X_VAL = df_VAL.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_VAL = df_VAL['mean_speed_wpau_v1']



# %% APPLY THE MODEL
X = data_set_with_pau.drop(['x_wp', 'y_wp','x_wop','y_wop', 'section', 'sample_id', 'region_id', 'speaker_id'], axis=1)
y = data_set_with_pau['y_wp']   

# %%
model = linear_model.LinearRegression(positive=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Score
r2 = model.score(X_test, y_test)
# %%
print(r2)
# %% 
X = data_set_with_pau.filter(regex='mean_feature_softmax')

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = model.score(X_test, y_test)
print(r2)

# %%
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


# %%
features = [A,B, C, D]

X = data_set_with_pau
y = data_set_with_pau['y']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

N=3

MSE_features_wpau = np.zeros(len(features))
scores_wpau = np.zeros(len(features))
for j in range(N):
    for i in range(len(features)):
        print('Features:', features[i])
        X_train_fi = X_train[features[i]]
        X_test_fi = X_test[features[i]]
        
        # Regression
        positive=True
        model = linear_model.LinearRegression(positive=positive)
        model.fit(X_train_fi, y_train)
        y_pred = model.predict(X_test_fi)
        MSE_features_wpau[i] += mean_squared_error(y_test, y_pred)
        scores_wpau[i] += model.score(X_test_fi, y_test)

mean_score_features_wpau = scores_wpau/N
mean_MSE_features_wpau = MSE_features_wpau/N

# %%
x = np.arange(len(features))
width = 0.35  # the width of the bars
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, mean_score_features_wpau, width, label='With pauses v1')
plt.xticks(np.arange(len(features)), ['A, dim:' + str(len(A)),'B, dim:'+ str(len(B)),'C, dim:'+ str(len(C)), 'D, dim:'+ str(len(D))]
           , rotation=0)

# Add in this plot the name of each feature
plt.title('Prediction of speed for any time')
plt.xlabel('Groups of features')
plt.ylabel('Mean R2 - '+ str(N) + ' iterations')
plt.legend()
plt.savefig('a_b_c_d_barplot.png')
plt.show()
# %%

# HI