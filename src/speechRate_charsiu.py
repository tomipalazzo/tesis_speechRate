

#%% TO DO 
# 1. Identify the silence and call it 'sil'
# 2. Do the Pablo's features
 


#%% Importing libraries
import sys
#import tables_speechRate as my_tables # ERROR QUE HAY QUE CORREGIR

#sys.path.insert(0,'../charsiu/src/')
from charsiu.src.Charsiu import Wav2Vec2ForFrameClassification, CharsiuPreprocessor_en, charsiu_forced_aligner
import torch 
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from datasets import load_dataset
import pandas as pd
import random
import librosa
from charsiu.src.utils import seq2duration, forced_align

# %%

TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

sample = TIMIT_train[1]# %%
#%%

path = '/home/tomi/Documents/tesis_speechRate/timit/data/TRAIN/DR1/FCJF0/SI648.WAV.wav'
#%% open txt
with open('/home/tomi/Documents/tesis_speechRate/timit/data/TRAIN/DR1/FCJF0/SI648.phn') as f:
    lines = f.readlines()
    for line in lines:
        print(line)
import os
current_directory = os.getcwd()
print(current_directory)
os.path.exists(path)
#%%
# if there are errors importing, uncomment the following lines and add path to charsiu
# import sys
# sys.path.append('path_to_charsiu/src')

# initialize model
charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')
# perform forced alignment

phonemes_index = np.arange(0,42)

phonemes = [charsiu.charsiu_processor.mapping_id2phone(int(i)) for i in phonemes_index]
print(phonemes)

alignment = charsiu.align(audio=path,
                          text='She had your dark suit in greasy wash water all year.')
# perform forced alignment and save the output as a textgrid file
charsiu.serve(audio=path,
              text='A sailboat may have a bone in her teeth one minute and lie becalmed the next.',
              save_to='ejemplo.TextGrid')
#%% Show the sample
# Access the audio data and sample rate
audio_data = sample['audio']['array']
sample_rate = sample['audio']['sampling_rate']

# Play the audio
ipd.Audio(audio_data, rate=sample_rate)

# %% Magic happens here


# Line 1: Instantiate a forced aligner object from the 'charsiu' library using a specified model.
# 'aligner' specifies the model to be used for alignment, likely based on the Wav2Vec 2.0 model trained for frame classification every 10 milliseconds.
charsiu = charsiu_forced_aligner(aligner='charsiu/en_w2v2_fc_10ms')

# Line 2: Load a pre-trained model from Hugging Face's 'transformers' library.
# This model is likely a fine-tuned version of Wav2Vec 2.0 for the task of frame classification, useful for tasks like forced alignment or phoneme recognition.
modelo = Wav2Vec2ForFrameClassification.from_pretrained("charsiu/en_w2v2_fc_10ms")
#%%
modelo = charsiu.aligner


#%%
# Line 3: Set the model to evaluation mode. This disables training specific behaviors like dropout, 
# ensuring the model's inference behavior is consistent and deterministic.
modelo.eval()

# Line 4: Instantiate a preprocessor for English from the 'charsiu' library.
# This object is likely used to prepare audio data by normalizing or applying necessary transformations 
# before it can be inputted to the model.
procesador = CharsiuPreprocessor_en()

#%%
phones, words = charsiu.charsiu_processor.get_phones_and_words('She had your dark suit in greasy wash water all year.')
phone_ids = charsiu.charsiu_processor.get_phone_ids(phones)
print(charsiu.charsiu_processor.sil_idx)
#%% GET PHONE IDs


#%%
# Line 5: Convert the audio data from a sample dictionary to a Torch tensor, necessary for processing with PyTorch models.
# 'np.array([sample['audio']['array']])' converts the audio samples to a NumPy array and wraps it in another array to add a batch dimension.
# '.astype(np.float32)' ensures that the data type is float32, which is typically required for neural network inputs in PyTorch.
x = torch.tensor(np.array([sample['audio']['array']]).astype(np.float32))

# Line 6: This line is inside a 'with' statement that disables gradient computation.
# 'torch.no_grad()' is crucial during inference to reduce memory consumption and speed up computations since backpropagation (gradient calculations) is not needed.
with torch.no_grad():
    # Line 7: Pass the preprocessed audio tensor 'x' through the model to obtain logits.
    # Logits are raw, non-normalized scores outputted by the last layer of a neural network. These need to be passed through a softmax layer to turn them into probabilities if necessary.
    predictions = modelo(x)
    y = predictions.logits
    y_mod = modelo(x)
    y_prob = torch.nn.functional.softmax(y, dim=-1)
    
#%% Information of predictions
print(predictions)

print(y)


y = y.numpy()[0].T
#y_prob = y_prob.numpy()[0].T
#print(y_mod)

#%%


#%%

plt.pcolor(y)
plt.colorbar()

plt.yticks(np.arange(0.5, len(phonemes)), phonemes)
plt.title('Phonogram')

#%% 
plt.pcolor(y_prob)
plt.colorbar()
plt.title('Phonogram with probabilities')


# %%
plt.hist(y[6,:])
# %% Choose randomly a subset of Train set and Test set

# Add seed for reproducibility
random.seed(42)
subset_train = random.sample(range(0, TIMIT_train.num_rows), 5)
subset_test = random.sample(range(0, TIMIT_test.num_rows), 2)

# %% now make it a function
def get_phonograms(dataset, model, subset):
    phonograms = []
    for i in range(len(subset)):
        sample = dataset[subset[i]]
        audio_data = sample['audio']['array']
        x = torch.tensor(np.array([audio_data]).astype(np.float32))
        with torch.no_grad():
            y = model(x).logits
        y = y.numpy()[0].T
        phonograms.append(y)
    return phonograms
# %%
phonograms = get_phonograms(TIMIT_train, modelo, subset_train)

#%%

plt.pcolor(phonograms[0])
plt.colorbar()

#%% Get the delta of the phonograms with Librosa
def get_delta(phonograms):
    deltas = []
    for i in range(len(phonograms)):
        delta = librosa.feature.delta(phonograms[i])
        deltas.append(delta)
    return deltas

deltas = get_delta(phonograms)
d_deltas = get_delta(deltas)
#%% 
plt.pcolor(phonograms[0])
plt.colorbar()

#%% 
plt.pcolor(d_deltas[0])
plt.colorbar()


#%% 
#%% Create DataFrame with the mean, std,  of each phone
features = []

for i in range(len(phonograms[0])):
    features.append({'mean_phonogram': np.mean(phonograms[0][i,:]),
                                'std_phonogram': np.std(phonograms[0][i,:]),
                                'mean_delta': np.mean(deltas[0][i,:]),
                                'std_delta': np.std(deltas[0][i,:]),
                                'mean_d_delta': np.mean(d_deltas[0][i,:]),
                                'std_d_delta': np.std(d_deltas[0][i,:]),
                                'abs_mean_phonogram': np.mean(np.abs(phonograms[0][i,:])),
                                'abs_mean_delta': np.mean(np.abs(deltas[0][i,:])),
                                'abs_mean_d_delta': np.mean(np.abs(d_deltas[0][i,:]))})
features = pd.DataFrame(features)
features = pd.DataFrame(features)
features.head()

# %% summary of one phonogram: max min mean std
features.describe()
# %%
features.shape
# %%
