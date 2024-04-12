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

def show_words_duration(sample, SR=16000):
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


def mean_speed_by(sample, SR=16000, phone=True):
    if phone:
        data = sample['phonetic_detail']['utterance']
        start = sample['phonetic_detail']['start']
        stop = sample['phonetic_detail']['stop']
    else:
        data = sample['word_detail']['utterance']
        start = sample['word_detail']['start']
        stop = sample['word_detail']['stop']

    time_of_data = np.zeros(stop[-1])
    data_interval = np.zeros(len(data))
    speed_of_data = np.zeros(len(data))
    amount_of_time = stop[-1]

    for i in range(len(data)):
        data_interval[i] = stop[i] - start[i]
    
    speed_of_data = 1 /(data_interval / SR)
    mean_speed = np.mean(speed_of_data)
    return mean_speed


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
    plt.title('Word Duration in Time Domain')

    return time, time_of_phone

#%%

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
# %% AUX functions

# It ignores the h# phones
def duration(x):
    return (x.iloc[-1]["start"]-x.iloc[1]["start"])/SR

def mean_speed(x):
    data = x['utterance']
    start = x['start']
    stop = x['stop']

    data_interval = np.zeros(len(data))
    speed_of_data = np.zeros(len(data))

    for i in range(len(data)):
        data_interval[i] = stop[i] - start[i]
    
    speed_of_data = 1 /(data_interval / SR)

    if(data[0] == 'pau'):
        speed_of_data[0] = 0

    mean_speed = np.mean(speed_of_data)
    return mean_speed

# %%
