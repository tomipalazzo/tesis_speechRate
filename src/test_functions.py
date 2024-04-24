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
import unittest

#%% 
# Load the TIMIT dataset from a specific directory
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%% Global variables
SR = 16000
SAMPLE = TIMIT_train[500]

#%%

class Test_mean_speed(unittest.TestCase):
    def test_only_silences(self):
        sample = {'utterance': ['h#', 'pau', 'pau'], 'start': [0, 1, 2], 'stop': [1, 2, 3]}
        assert fn.mean_speed(sample) == 0

# %% run tests
if __name__ == '__main__':
    unittest.main()

# %%
