#%% Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import numpy as np
import statsmodels.api as sm
import random
import utils as ut
from datasets import load_dataset
import time

#%%
# Load the TIMIT dataset from a specific directory
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Global variables
SR = 16000
SAMPLE = TIMIT_train[0]

#%% Show the sample
# Access the audio data and sample rate
audio_data = SAMPLE['audio']['array']
sample_rate = SAMPLE['audio']['sampling_rate']

# Play the audio
ipd.Audio(audio_data, rate=sample_rate)

#%%
# Print the phones
phones = SAMPLE['phonetic_detail']
print(phones)

#%%
# Check if all the sampling rates are measured at 16000 Hz
ALL_EQUAL_SAMPLING_RATE = True
DIFFERENTS_SAMP_RATE = []
for i in range(TIMIT_train.num_rows):
    sample = TIMIT_train[i]
    is_equal = sample['audio']['sampling_rate'] == SR
    if not is_equal:
        DIFFERENTS_SAMP_RATE.append(sample['audio']['sampling_rate'])
        ALL_EQUAL_SAMPLING_RATE = False

if(ALL_EQUAL_SAMPLING_RATE): 
    show = 'All the samples has the same sampling rate (16000 per second)'
else:
    show = 'There are different sampling rates in the dataset'
print(show)


#%% Show words duration
ut.show_words_duration(sample=SAMPLE, SR=SR)

#%% SPEED BY phone
X, y = ut.speed_by_phone(sample=SAMPLE)

#%% SPEED REGRESSION
ut.speed_smoothed_regression(X=X, y=y, bandwidth=0.01)

#%% Create the dataframe with the information of each record

t0 = time.time()
TIMIT_df = ut.TIMIT_df_by_record()   
TIMIT_df.build_phone_test(TIMIT_test)
TIMIT_df.build_phone_train(TIMIT_train)
TIMIT_df.build_word_test(TIMIT_test)
TIMIT_df.build_word_train(TIMIT_train)
t1 = time.time()
print('Time: ', t1-t0)

#%% Print the first 5 rows of the test set
TIMIT_df.phone_test.head()

# %% Print the first 5 rows of the silence records
TIMIT_df.phone_test[TIMIT_df.phone_test['phone_rate'] == 0]
# %% 
TIMIT_df.phone_test.groupby("sample_id")["duration_s"].sum()

# %% Make a DF with the information of the samples

df_bySample_train = ut.TIMIT_df_by_sample(TIMIT_df.phone_train)

# %% Print the first 5 rows of the samples
df_bySample_train.head()

#%% Show the distribution of the duration of the samples
sample1 = TIMIT_df.phone_train[TIMIT_df.phone_train['sample_id'] == 'DR1_CJF0_SA1']

#%% This block try to do a plot of the phone rate of the sample.
size = sample1['stop'][len(sample1)-1]
time1 = np.arange(size)/SR
phone_rate_axis = np.zeros(size)
phones = sample1.shape[0]

start=time.time()
for i in range(phones):
    phone_rate = sample1['phone_rate'][i]
    phone_rate_axis[sample1['start'][i]:sample1['stop'][i]] = phone_rate

end = time.time()
print('Time: ', end-start)

# %% Instantaneous speed vs mean speed
ut.speed_smoothed_regression(X=time1, y=phone_rate_axis, bandwidth=0.01, mean_speed=mean_phone_rate)

# %%
