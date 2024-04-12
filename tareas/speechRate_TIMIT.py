#%% Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import numpy as np
import statsmodels.api as sm
import random
import functions as fn
from datasets import load_dataset


#%%
# Load the TIMIT dataset from a specific directory
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Global variables
SR = 16000
SAMPLE = TIMIT_train[500]


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
fn.show_words_duration(sample=SAMPLE, SR=SR)

#%% SPEED BY WORD

X, y = fn.speed_by_phone(sample=SAMPLE)
X.shape[0]
# %% SPEED BY PHONE
fn.speed_by_phone(sample=SAMPLE)


#%% SPEED REGRESSION
fn.speed_smoothed_regression(X=X, y=y, bandwidth=0.01)
# %% All the information in one dataSet
# Idea: make a DF of all the PHONES and add columns with important information

TIMIT_test_phones_df = []
for sample in TIMIT_test:
    sample_id = sample['dialect_region'] + '_' + sample['speaker_id'] + '_' + sample['id']
    dataframe = pd.DataFrame(sample['phonetic_detail'])
    dataframe["duration_s"]=(dataframe["stop"]-dataframe["start"])/SR
    dataframe["phone_rate"] = 1/dataframe["duration_s"] 
    dataframe['sample_id'] = sample_id
    

    TIMIT_test_phones_df.append(dataframe)

TIMIT_test_phones_df = pd.concat(TIMIT_test_phones_df)

# %% All the information in one dataSet
# Idea: make a DF of all the utterances add columns with important information

TIMIT_test_words_df = []
for sample in TIMIT_test:
    sample_id = sample['dialect_region'] + '_' + sample['speaker_id'] + '_' + sample['id']
    dataframe = pd.DataFrame(sample['word_detail'])
    dataframe["duration_s"]=(dataframe["stop"]-dataframe["start"])/SR
    dataframe["phone_rate"] = 1/dataframe["duration_s"] 
    dataframe['sample_id'] = sample_id
    

    TIMIT_test_words_df.append(dataframe)

TIMIT_test_words_df = pd.concat(TIMIT_test_words_df)


# %%
TIMIT_test_phones_df.head()
# %% 
TIMIT_test_phones_df.groupby("sample_id")["duration_s"].sum()

# %% Make a DF with the information of the samples
TIMIT_test_df_samples = pd.DataFrame()
TIMIT_test_df_samples["duration_wpau"]=TIMIT_test_phones_df.groupby("sample_id").apply(duration)
TIMIT_test_df_samples["mean_speed"]=TIMIT_test_phones_df.groupby("sample_id").apply(mean_speed)

# %%
TIMIT_test_df_samples.head()

#%%

def aux(x):
    return (x.iloc[-1]["start"]-x.iloc[1]["start"])/SR

#%%
