#%% Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import numpy as np
import statsmodels.api as sm
import random
from datasets import load_dataset


#%%
# Load the TIMIT dataset from a specific directory
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Global variables
SR = 16000
SAMPLE = TIMIT_train[500]

#%%
# Select a sample from the training set


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

def show_words_duration(sample):
    words = sample['word_detail']['utterance']
    start = sample['word_detail']['start']
    stop = sample['word_detail']['stop']

    words.insert(0, 'silence')
    stop.insert(0, start[0])
    time_of_word = []

    i = 0
    for j in range(stop[-1]):
        i_am_in_the_border = j >= stop[i]
        if i_am_in_the_border:
            i += 1
        time_of_word.append(words[i])

    # Plot using frequency as the x-axis
    time = np.arange(stop[-1]) / SR
    plt.plot(time, time_of_word)
    plt.xlabel('Time (s)')
    plt.ylabel('Words')
    plt.title('Word Duration in Time Domain')
    plt.show()

#%%
show_words_duration(sample=SAMPLE)

# %%

def speed_by_word(sample, SR=16000):
    words = sample['word_detail']['utterance']
    start = sample['word_detail']['start']
    stop = sample['word_detail']['stop']
    time_of_word = np.zeros(stop[-1])
    word_interval = np.zeros(len(words))
    speed_of_word = np.zeros(len(words))
    amount_of_time = stop[-1]



    for i in range(len(words)):
        word_interval[i] = stop[i] - start[i]
        

    
    speed_of_word = 1 /(word_interval / SR)
    print(speed_of_word)
    i = 0
    for j in range(start[0], stop[-1]):
        i_am_in_the_border = j >= stop[i]
        if i_am_in_the_border:
            i += 1
        time_of_word[j] = speed_of_word[i]

    
    # Plot using time as the x-axis
    time = np.arange(amount_of_time) / SR
    plt.plot(time, time_of_word)
    plt.xlabel('Time (s)')
    plt.ylabel('Words')
    plt.title('Word Duration in Time Domain')

    return time, time_of_word
# %%
def speed_by_phone(sample, SR=16000):

    phones = sample['phonetic_detail']['utterance']
    start = sample['phonetic_detail']['start']
    stop = sample['phonetic_detail']['stop']
    time_of_phone = np.zeros(stop[-1])
    phone_interval = np.zeros(len(phones))
    speed_of_phone = np.zeros(len(phones))
    amount_of_time = stop[-1]

    for i in range(len(phones)):
        phone_interval[i] = stop[i] - start[i]
        

    
    speed_of_phone = 1 /(phone_interval / SR)
    print(speed_of_phone)
    i = 0
    for j in range(start[0], stop[-1]):
        i_am_in_the_border = j >= stop[i]
        if i_am_in_the_border:
            i += 1
        time_of_phone[j] = speed_of_phone[i]

    
    # Plot using time as the x-axis
    time = np.arange(amount_of_time) / SR
    plt.plot(time, time_of_phone)
    plt.xlabel('Time (s)')
    plt.ylabel('Words')
    plt.title('Word Duration in Time Domain')

    return time, time_of_phone

#%%

X, y = speed_by_phone(sample=SAMPLE)
X.shape[0]
# %%
speed_by_phone(sample=SAMPLE)


#%% SPEED REGRESSION
def speed_smoothed_regression(X, y, bandwidth=0.1):
    
    # if data is large, subsample
    max_length = 10000
    if(len(X) > max_length):
        id = np.arange(0,X.shape[0],10)
        X = X[id]
        y = y[id]

    # Add a constant to X for the regression model
    X = sm.add_constant(X)
    y = sm.add_constant(y)

    # Fit the Nadaraya-Watson kernel regression model with the specified bandwidth
    model = sm.nonparametric.KernelReg(endog=y[:,1], exog=X[:, 1], var_type='c', reg_type='lc', bw=[bandwidth])
    y_pred, y_std = model.fit(X[:, 1])



    # Plot the data and the regression line
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 1], y[:,1], alpha=0.5, label='Data')
    plt.plot(X[:, 1], y_pred, color='red', label='Nadaraya-Watson Kernel Regression')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Nonparametric Regression with Specified Bandwidth')
    plt.legend()
    plt.show()

#%%
speed_smoothed_regression(X=X, y=y, bandwidth=0.01)
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
# Idea: make a DF of all the WORDS and add columns with important information

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

# %%
# for k,g in 

def fn(x):
    return (x.iloc[-1]["start"]-x.iloc[1]["start"])/SR

TIMIT_test_df_samples = pd.DataFrame()
TIMIT_test_df_samples["duration_wpau"]=TIMIT_test_phones_df.groupby("sample_id").apply(fn)
# %%


def fn(x):
    return (x.iloc[-1]["start"]-x.iloc[1]["start"])/SR

#%%

