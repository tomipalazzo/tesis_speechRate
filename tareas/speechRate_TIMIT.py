#%%
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import numpy as np
import random
from datasets import load_dataset

#%%
# Load the TIMIT dataset from a specific directory
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%%
# Local variables
SR = 16000

#%%
# Select a sample from the training set
sample = TIMIT_train[5]

# Access the audio data and sample rate
audio_data = sample['audio']['array']
sample_rate = sample['audio']['sampling_rate']

# Play the audio
ipd.Audio(audio_data, rate=sample_rate)

#%%
# Print the phones
phones = sample['phonetic_detail']
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

print(ALL_EQUAL_SAMPLING_RATE)

#%%
# Plot the duration of each word for one sample
sample = TIMIT_train[0]
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

#%%
# Plot using frequency as the x-axis
freq = np.arange(stop[-1])
plt.plot(freq, time_of_word)
plt.xlabel('Sample')
plt.ylabel('Words')
plt.title('Word Duration in Sample Domain')
plt.show()

#%%
# Plot using time as the x-axis
time = np.arange(stop[-1]) / SR
plt.plot(time, time_of_word)
plt.xlabel('Time (s)')
plt.ylabel('Words')
plt.title('Word Duration in Time Domain')
plt.show()

# %%

def speed_by_word(sample, SR=16000):

    sample = TIMIT_train[0]
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

    sample = TIMIT_train[0]
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

X, y = speed_by_word(sample=sample)
X.shape[0]
# %%
speed_by_phone(sample=sample)


#%% SPEED REGRESSION





#%%

import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Generate sample data
id = np.arange(0,X.shape[0],10)


X_sample = X[id]
y_sample = y[id]

# Add a constant to X for the regression model
X_sample = sm.add_constant(X_sample)
y_sample = sm.add_constant(y_sample)


#%%
# Fit the Nadaraya-Watson kernel regression model with the specified bandwidth
model = sm.nonparametric.KernelReg(endog=y_sample[:,1], exog=X_sample[:, 1], var_type='c', reg_type='lc', bw=[bandwidth])
y_pred, y_std = model.fit(X_sample[:, 1])



# Plot the data and the regression line
plt.figure(figsize=(10, 6))
plt.scatter(X_sample[:, 1], y_sample[:,1], alpha=0.5, label='Data')
plt.plot(X_sample[:, 1], y_pred, color='red', label='Nadaraya-Watson Kernel Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Nonparametric Regression with Specified Bandwidth')
plt.legend()
plt.show()


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

