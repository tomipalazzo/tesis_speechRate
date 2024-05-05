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

SR = 16000

def plot_words_duration(sample, SR=16000):

    '''
    This function plots the duration of each word in the sample.

    plot_words_duration(sample, SR=16000)
    Parameters: sample - The sample from the TIMIT dataset.
                SR - The sample rate of the audio.
    Output: A plot with the duration of each word in the sample.
    '''


    words = sample['word_detail']['utterance'].copy()
    start = sample['word_detail']['start'].copy()
    stop = sample['word_detail']['stop'].copy()

    words.insert(0, 'silence')
    stop.insert(0, start[0])
    start.insert(0, 0)
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
def speed_by_word(sample, SR=16000):

    '''
    This function plots the speed of each word in the sample.

    speed_by_word(sample, SR=16000)
    Parameters: sample - The sample from the TIMIT dataset.
                SR - The sample rate of the audio.
    Output: A plot with the speed of each word in the sample.
    '''

    words = sample['word_detail']['utterance'].copy()
    start = sample['word_detail']['start'].copy()
    stop = sample['word_detail']['stop'].copy()
    
    words.insert(0, 'silence')
    stop.insert(0, start[0])
    start.insert(0, 0)
 
    
    time_of_word = np.zeros(stop[-1])
    word_interval = np.zeros(len(words))
    speed_of_word = np.zeros(len(words))
    amount_of_time = stop[-1]



    for i in range(len(words)):
        word_interval[i] = stop[i] - start[i]
        

    
    speed_of_word = 1 /(word_interval / SR)
    #speed_by_word[0] = 0 # Is the silence
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
    plt.title('Word Speed in Time Domain')

    return time, time_of_word


def mean_speed_by(sample, SR=16000, phone=True):

    '''
    This function calculates the mean speed of the sample.

    mean_speed_by(sample, SR=16000, phone=True)

    Parameters: sample - The sample from the TIMIT dataset.
                SR - The sample rate of the audio.
                phone - If True, the function calculates the speed by phone. If False, the function calculates the speed by word.
    Output: The mean speed of the sample.

    '''


    if phone:
        data = sample['phonetic_detail']['utterance']
        start = sample['phonetic_detail']['start']
        stop = sample['phonetic_detail']['stop']
    else:
        data = sample['word_detail']['utterance']
        start = sample['word_detail']['start']
        stop = sample['word_detail']['stop']

    data_interval = np.zeros(len(data))
    speed_of_data = np.zeros(len(data))

    for i in range(len(data)):
        data_interval[i] = stop[i] - start[i]
    
    speed_of_data = 1 /(data_interval / SR)
    mean_speed = np.mean(speed_of_data)
    return mean_speed


def plot_speed_by_phone(sample, SR=16000):

    '''
    This function plots the speed of each phone in the sample.
    
    speed_by_phow(sample, SR=16000)

    Parameters: sample - The sample from the TIMIT dataset.
                SR - The sample rate of the audio.
    Output: A plot with the speed of each phone in the sample.
    '''

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
    
    for i in range(len(phones)):
        if phones[i] == 'pau':
            speed_of_phone[i] = 0

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
    plt.title('Phone Speed in Time Domain')

    return time, time_of_phone

#%%

def speed_smoothed_regression(X, y, bandwidth=0.1, mean_speed=0):
    
    '''
    This function plots the speed using a smoothed regression..

    speed_smoothed_regression(X, y, bandwidth=0.1)
    Parameters: X - The data.
                y - The target.
                bandwidth - The bandwidth of the regression.
    Output: A plot with the speed using a smoothed regression.

    '''


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

    if mean_speed != 0:
        plt.axhline(y=mean_speed, color='r', linestyle='dashed', label='Mean Speed = {}'.format(mean_speed))

    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('Nonparametric Regression with Specified Bandwidth')
    plt.legend()
    plt.show()
# %% AUX functions

# It ignores the h# phones  
def duration(x, DF=16000):
    return (x.iloc[-1]["start"]-x.iloc[1]["start"])/SR

# phones: pau = epi = h# = 0
def mean_speed(x):

    silence = ['pau', 'epi', 'h#']
    # data = x['utterance']
    # start = x['start']
    # stop = x['stop']

    # data_interval = np.zeros(len(data))
    # speed_of_data = np.zeros(len(data))

    # data_interval = stop - start
    
    # speed_of_data = 1 /(data_interval / SR)

    
    # for i in range(len(data)):

    #     if(data[i] in silence):
    #         speed_of_data[i] = 0

    # mean_speed = np.mean(speed_of_data)

    # ## NOTA: tiene que devolver 2 valores, el mean_speed y el mean_speed sin los silencios

    avg_dur_wopau = x.loc[~x["utterance"].isin(silence),:]["duration_s"].mean()
    avg_speed_wopau = 1/avg_dur_wopau

    
    sum_dur_wopau = x.loc[~x["utterance"].isin(silence),:]["duration_s"].sum() # Excluding the silence
    avg_dur_wpau = sum_dur_wopau/len(x.iloc[1:-1]) # Excluding the h# at the beginning and the end
    avg_speed_wpau = 1/avg_dur_wpau
        
    return avg_dur_wopau, avg_speed_wopau, avg_dur_wpau, avg_speed_wpau

def avg_speed_wopau(x):    
    return mean_speed(x)[1]

def avg_speed_wpau(x):
    return mean_speed(x)[3]

# %%

class TIMIT_df_by_record:

    '''
    This class builds a dataframe with the information of the samples in the TIMIT dataset.

    Atributes: phone_train - A list with the information of the phones in the training set.
                phone_test - A list with the information of the phones in the test set.
                word_train - A list with the information of the words in the training set.
                word_test - A list with the information of the words in the test set.
    Methods: build_phone_test - Builds the phone_test list.
            build_phone_train - Builds the phone_train list.
            build_word_test - Builds the word_test list.
            build_word_train - Builds the word_train list.
    '''
    def __init__(self):
        self.phone_train = []
        self.phone_test = []
        self.word_train = []
        self.word_test = []
    

    def build_phone_test(self, TIMIT_test):
        for sample in TIMIT_test:
            sample_id = sample['dialect_region'] + '_' + sample['speaker_id'] + '_' + sample['id']
            dataframe = pd.DataFrame(sample['phonetic_detail'])
            dataframe["duration_s"]=(dataframe["stop"]-dataframe["start"])/SR
            dataframe["phone_rate"] = 1/dataframe["duration_s"] 
            
            
            # If phone[i] in silence then phone_rate = 0
            silence = ['h#', 'pau', 'epi']
            dataframe['phone_rate'] = dataframe['phone_rate'].where(~dataframe['utterance'].isin(silence), 0)


            
            dataframe['sample_id'] = sample_id
            self.phone_test.append(dataframe)
    
        self.phone_test = pd.concat(self.phone_test)


    def build_phone_train(self, TIMIT_train):
        for sample in TIMIT_train:
            sample_id = sample['dialect_region'] + '_' + sample['speaker_id'] + '_' + sample['id']
            dataframe = pd.DataFrame(sample['phonetic_detail'])
            dataframe["duration_s"]=(dataframe["stop"]-dataframe["start"])/SR
            dataframe["phone_rate"] = 1/dataframe["duration_s"] 

            # If phone[i] in silence then phone_rate = 0

            silence = ['h#', 'pau', 'epi']
            dataframe['phone_rate'] = dataframe['phone_rate'].where(~dataframe['utterance'].isin(silence), 0)


            dataframe['sample_id'] = sample_id
            self.phone_train.append(dataframe)
        self.phone_train = pd.concat(self.phone_train)
    
    def build_word_test(self, TIMIT_test):
        for sample in TIMIT_test:
            sample_id = sample['dialect_region'] + '_' + sample['speaker_id'] + '_' + sample['id']
            dataframe = pd.DataFrame(sample['word_detail'])
            dataframe["duration_s"]=(dataframe["stop"]-dataframe["start"])/SR
            dataframe["phone_rate"] = 1/dataframe["duration_s"] 
            dataframe['sample_id'] = sample_id
            self.word_test.append(dataframe)
    
        self.word_test = pd.concat(self.word_test)
    
    def build_word_train(self, TIMIT_train):
        for sample in TIMIT_train:
            sample_id = sample['dialect_region'] + '_' + sample['speaker_id'] + '_' + sample['id']
            dataframe = pd.DataFrame(sample['word_detail'])
            dataframe["duration_s"]=(dataframe["stop"]-dataframe["start"])/SR
            dataframe["phone_rate"] = 1/dataframe["duration_s"] 
            dataframe['sample_id'] = sample_id
            self.word_train.append(dataframe)
        self.word_train = pd.concat(self.word_train)
        


#%%

def TIMIT_df_by_sample_phones(df_by_record_of_phones):
    TIMIT_df_samples = pd.DataFrame()
    TIMIT_df_samples["duration_wpau"] = df_by_record_of_phones.groupby("sample_id").apply(duration)
    TIMIT_df_samples["mean_speed_wpau"] = df_by_record_of_phones.groupby("sample_id").apply(avg_speed_wpau)
    TIMIT_df_samples["mean_speed_wopau"] = df_by_record_of_phones.groupby("sample_id").apply(avg_speed_wopau)
    return TIMIT_df_samples
# %%
