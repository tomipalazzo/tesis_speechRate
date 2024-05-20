#%% Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import numpy as np
import statsmodels.api as sm
import random
import src.utils as ut
from datasets import load_dataset
import time

# %% The idea of this file is show the behavior of the metrics 
#     in our speech rate analysis   

TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Global variables
SR = 16000
SAMPLE = TIMIT_train[0]

#%% # Print the phones
phones = SAMPLE['phonetic_detail']
print(phones)

#%% plot words duration
#ut.plot_words_duration(sample=SAMPLE, SR=SR)


#%% SPEED BY phone
#X, y = ut.plot_speed_by_phone(sample=SAMPLE)

#%% SPEED REGRESSION
#ut.speed_smoothed_regression(X=X, y=y, bandwidth=0.01)

#%% Create the dataframe with the information of each record


record_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_test.csv')
record_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_train.csv')
sample_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_train.csv')
sample_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_test.csv')


#%%
#filter the DR1_CJF0_SA1
SAMPLE = record_phone_train[record_phone_train['sample_id'] == 'DR1_CJF0_SA1']


# %% Load the dataset
def m1(phones, step_size=10, window_size=100, with_pau = False):
    """
    This function makes a windowed regression of the data X and y. The window size is the number of points that the 
    regression will take into account. The step size is the number of points that the window will move in each iteration.
    """
    # Initialize the variables
    start = phones['start']
    stop = phones['stop']
    n = start.shape[0]
    T = stop[n-1]   # Length of the audio
    steps = np.arange(0,T,step_size)
    y_hat = np.zeros(len(steps))

    # Indicators
    how_many_completely = 0
    how_many_partialy = 0
    how_many_out = 0

    silence = ['h#', 'pau', 'epi']
    for i in range(len(steps)):
        left = max(0, steps[i]-window_size/2)
        right = min(T-1,steps[i]+window_size/2)

        for phone_j in range(n):


            onset = start[phone_j]
            is_silence = phones['utterance'][phone_j] in silence
            if (not is_silence) or (with_pau):

    #                is_completely_in_window = (start[phone_i] > left) and (stop[phone_i] < right)
    #                is_completely_out_of_window = (start[phone_i]> right) or (stop[phone_i] < left)
    #                is_completely_out_of_window = True
    #                if is_completely_in_window:
    #                    y_hat[i] += 1/2
    #                    how_many_completely += 1
    #                elif not is_completely_out_of_window:
    #                    y_hat[i] += 1/2
    #                    how_many_partialy += 1
    #                else:
    #                    how_many_out += 1
                if onset > left and onset < right:
                    y_hat[i] += 1
                    

                



    return steps/16000, y_hat

# %%

def m2(phones, SR=16000):

    '''
    This function plots the speed of each phone in the sample.
    
    speed_by_phow(phones, SR=16000)

    Parameters: phones - The phones of the sample.
                SR - The sample rate of the audio.
    Output: A plot with the speed of each phone in the sample.
    '''
    n = len(phones)
    utterance = phones['utterance']
    start = phones['start']
    stop = phones['stop']
    time_of_phone = np.zeros(stop[n-1])
    phone_interval = np.zeros(len(phones))
    amount_of_time = stop[n-1]

    phone_interval = stop - start
        
    # NOTE try to do it more neatly

    speed_of_phone = 1 /(phone_interval / SR)    
    for i in range(len(utterance)):
        if utterance[i] == 'pau':
            speed_of_phone[i] = 0

    i = 0
    for j in range(start[0], stop[n-1]):
        i_am_in_the_border = j >= stop[i]
        if i_am_in_the_border:
            i += 1
        time_of_phone[j] = speed_of_phone[i]

    


    return time, time_of_phone

#%%
def m3(phones, t = 0, step = 0.1):
    if t == 0:
        t = phones['stop'][len(phones)-1]/16000
    
    t_range = np.arange(0,t,step)
    how_many_phones_before_t = np.zeros(len(t_range))
    speech_rate = np.zeros(len(t_range))

    for i in range(len(t_range)):
        for j in range(len(phones)):
            if phones['start'][j] < t_range[i]:
                how_many_phones_before_t[i] += 1
        speech_rate[i] = how_many_phones_before_t[i]/t_range[i]
    
    return t_range,speech_rate



#%%
def m4(phones, T=0, t_step = 0.1):


    start = phones['start']/16000
    stop = phones['stop']/16000

    
    n = len(start)

    if T == 0:
        T = stop[n-1]

    count = 0
    steps = np.arange(0,T+1,t_step)
    speech_rate = np.zeros(len(steps))
    value = 0

    phone_index = 0
    t = 1
    for i in range(1,len(steps)):

        if steps[i] > start[phone_index]:
            t = 1
            phone_index += 1
            speech_rate[i] = speech_rate[i-1] + 1
            value = speech_rate[i]
        else:
            t += t_step
            speech_rate[i] = value/np.exp(t)

        if phone_index == n:
            break
    return steps, speech_rate


def plot_metrics(x,y,name='plot',hiperparam = []):
    # Plot using time as the x-axis
    
    #hiperparam2string 
    hiperparam2string = ''
    for i in range(len(hiperparam)):
        hiperparam2string += str(hiperparam[i])
    
    name_file = name + hiperparam2string + '.png'

    plot = plt.figure()
    plt.plot(x,y)
    plt.title(name)
    plt.xlabel('Time (s)')
    plt.ylabel(name)
    plt.grid()
    plt.show()

    
#%% ============================ TEST METRICS ============================

# %% Test m1

step_size = 1/4     #seg
window_size = 1/4   #seg
with_pause = False

x1, y1 = m1(SAMPLE, step_size=step_size*1600, window_size=window_size*16000, with_pau=with_pause)
#%%
plot_metrics(x1,y1,'m1', [step_size, window_size, with_pause])

#%% Test m2
x2, y2 = m2(SAMPLE, SR=16000) 

plot_metrics(x2,y2,'m2.png')
#%% Test m3
x3,y3 = m3(SAMPLE, t=0, step = 1)

plot_metrics(x3,y3,'m3.png')






#%% Test m4
x, y = m4(SAMPLE, T=0, t_step=0.0001)

plot_metrics(x,y,'m4.png')
