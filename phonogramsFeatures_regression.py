#%% #import tables_speechRate as my_tables

import sys
#from charsiu.src import models
from charsiu.src.Charsiu import CharsiuPreprocessor_en, charsiu_forced_aligner
import torch 
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from datasets import load_dataset
import pandas as pd
import random
import librosa
import src.tables_speechRate as my_tables

#%%
# Load the dataset
sample_phone_train = my_tables.TIMIT_df_by_record.phone_train

# %%
sample_phone_train
# %% Magic happens here


# Line 1: Instantiate a forced aligner object from the 'charsiu' library using a specified model.
# 'aligner' specifies the model to be used for alignment, likely based on the Wav2Vec 2.0 model trained for frame classification every 10 milliseconds.
charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

# Line 2: Load a pre-trained model from Hugging Face's 'transformers' library.
# This model is likely a fine-tuned version of Wav2Vec 2.0 for the task of frame classification, useful for tasks like forced alignment or phoneme recognition.
modelo = Wav2Vec2ForFrameClassification.from_pretrained("charsiu/en_w2v2_fc_10ms")

# Line 3: Set the model to evaluation mode. This disables training specific behaviors like dropout, 
# ensuring the model's inference behavior is consistent and deterministic.
modelo.eval()

# Line 4: Instantiate a preprocessor for English from the 'charsiu' library.
# This object is likely used to prepare audio data by normalizing or applying necessary transformations 
# before it can be inputted to the model.
procesador = CharsiuPreprocessor_en()

# Line 5: Convert the audio data from a sample dictionary to a Torch tensor, necessary for processing with PyTorch models.
# 'np.array([sample['audio']['array']])' converts the audio samples to a NumPy array and wraps it in another array to add a batch dimension.
# '.astype(np.float32)' ensures that the data type is float32, which is typically required for neural network inputs in PyTorch.
x = torch.tensor(np.array([sample['audio']['array']]).astype(np.float32))

# Line 6: This line is inside a 'with' statement that disables gradient computation.
# 'torch.no_grad()' is crucial during inference to reduce memory consumption and speed up computations since backpropagation (gradient calculations) is not needed.
with torch.no_grad():
    # Line 7: Pass the preprocessed audio tensor 'x' through the model to obtain logits.
    # Logits are raw, non-normalized scores outputted by the last layer of a neural network. These need to be passed through a softmax layer to turn them into probabilities if necessary.
    y = modelo(x).logits
    y_prob = torch.nn.functional.softmax(y, dim=-1)


y = y.numpy()[0].T
y_prob = y_prob.numpy()[0].T


