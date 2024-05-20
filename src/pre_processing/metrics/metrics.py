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


# TODO Ignore the h# phones 

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
