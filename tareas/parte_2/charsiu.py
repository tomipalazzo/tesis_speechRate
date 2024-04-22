

#%% TO DO 
# 1. Check the values of mean_phonegram
# 2. use the Datasets os speechRate_TIMIT.py to get the phonograms
# 3. Add the 2 features that are missing in the phonograms dataframe
 


#%% Importing libraries
import sys
sys.path.insert(0,'../charsiu/src/')
from Charsiu import Wav2Vec2ForFrameClassification, CharsiuPreprocessor_en, charsiu_forced_aligner
import torch 
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from datasets import load_dataset
import pandas as pd
import random
import librosa

# %%

TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

sample = TIMIT_train[1]# %%


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


#%%

plt.pcolor(y)
plt.colorbar()
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



#%% Create DataFrame with the mean, std,  of each phone
features = []

for i in range(len(phonograms[0])):
    features.append({'mean_phonogram': np.mean(phonograms[0][i,:]),
                                'std_phonogram': np.std(phonograms[0][i,:]),
                                'mean_delta': np.mean(deltas[0][i,:]),
                                'std_delta': np.std(deltas[0][i,:]),
                                'mean_d_delta': np.mean(d_deltas[0][i,:]),
                                'std_d_delta': np.std(d_deltas[0][i,:])})
features = pd.DataFrame(features)
features = pd.DataFrame(features)
features.head()

# %% summary of one phonogram: max min mean std
features.describe()
# %%
features.shape
# %%
