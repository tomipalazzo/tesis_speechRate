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
fn.show_words_duration(sample=SAMPLE, SR=SR)

#%% SPEED BY phone
X, y = fn.speed_by_phone(sample=SAMPLE)


#%% SPEED REGRESSION
fn.speed_smoothed_regression(X=X, y=y, bandwidth=0.01)


#%% Test the class TIMIT_phones_df
from functions import TIMIT_df_by_record

t0 = time.time()
TIMIT_df = TIMIT_df_by_record()   
TIMIT_df.build_phone_test(TIMIT_test)
TIMIT_df.build_phone_train(TIMIT_train)
TIMIT_df.build_word_test(TIMIT_test)
TIMIT_df.build_word_train(TIMIT_train)
t1 = time.time()
print('Time: ', t1-t0)

#%% Print the first 5 rows of the test set
TIMIT_df.phone_test.head()

# %%
TIMIT_df.phone_test[TIMIT_df.phone_test['phone_rate'] == 0]
# %% 
TIMIT_df.phone_test.groupby("sample_id")["duration_s"].sum()

# %% Make a DF with the information of the samples
# import functions as fn
TIMIT_test_df_samples = pd.DataFrame()
TIMIT_test_df_samples["duration_wpau"]=TIMIT_df.phone_test.groupby("sample_id").apply(fn.duration) # Without begin/end marker
TIMIT_test_df_samples["mean_speed"]=TIMIT_df.phone_test.groupby("sample_id").apply(fn.mean_speed) # pau = epi = h# = 0



# %%
TIMIT_test_df_samples.head()

#%%
#%%
sample1 = TIMIT_phones_df.phone_test[TIMIT_phones_df.phone_test['sample_id'] == 'DR1_AKS0_SA1']

#%%
size = sample1['stop'][len(sample1)-1]
time1 = np.zeros(size)
phone_rate_axis = np.zeros(size)

j=0
for i in range(size):
    border = sample1['stop'][j]
    if i >= border:
        j+=1
    
    time1[i] = i/SR
    phone_rate_axis[i] = sample1['phone_rate'][j]


plt.plot(time1, phone_rate_axis)
     

#plt.scatter()
# %% Instantaneous speed vs mean speed
fn.speed_smoothed_regression(X=time1, y=phone_rate_axis, bandwidth=0.01, mean_speed =  TIMIT_test_df_samples['mean_speed'][0])

# %%
