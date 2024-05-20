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
N = 400
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
# %%
