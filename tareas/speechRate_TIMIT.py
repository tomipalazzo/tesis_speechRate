#%%
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import numpy as np
from datasets import load_dataset

#%%
# Load the TIMIT dataset from a specific directory
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%%
# Local variables
FREQUENCY = 16000

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
    is_equal = sample['audio']['sampling_rate'] == 16000
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
plt.xlabel('Frequency (Hz)')
plt.ylabel('Words')
plt.title('Word Duration in Frequency Domain')
plt.show()

#%%
# Plot using time as the x-axis
time = np.arange(stop[-1]) / FREQUENCY
plt.plot(time, time_of_word)
plt.xlabel('Time (s)')
plt.ylabel('Words')
plt.title('Word Duration in Time Domain')
plt.show()

# %%

def speed_by_word(sample, freq=16000):

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
        

    
    speed_of_word = 1 /(word_interval / 16000)
    print(speed_of_word)
    i = 0
    for j in range(start[0], stop[-1]):
        i_am_in_the_border = j >= stop[i]
        if i_am_in_the_border:
            i += 1
        time_of_word[j] = speed_of_word[i]

    
    # Plot using time as the x-axis
    time = np.arange(amount_of_time) / FREQUENCY
    plt.plot(time, time_of_word)
    plt.xlabel('Time (s)')
    plt.ylabel('Words')
    plt.title('Word Duration in Time Domain')
# %%
speed_by_word(sample=sample)
# %%
