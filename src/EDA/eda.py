#%%
import sys
#from charsiu.src import models
from charsiu.src.Charsiu import Wav2Vec2ForFrameClassification, CharsiuPreprocessor_en, charsiu_forced_aligner, charsiu_chain_attention_aligner, charsiu_chain_forced_aligner, charsiu_predictive_aligner
import torch 
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from datasets import load_dataset
import pandas as pd
import random
import librosa
#import src.tables_speechRate as my_tables
import src.utils as ut
import time
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import seaborn as sns
import ast  # For safely evaluating strings containing Python literals
from src.pre_processing.metrics.metrics import m1


# %%
# Idea:
# - Boxplot of different speakers
# - PCA in 1 dim of the data which explains the reason of approximate the problem with linear model


#%% Load the data
# Load the data
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

# %%
record_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_test.csv')
record_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_record_phone_train.csv')
sample_phone_train = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_train.csv')
sample_phone_test = pd.read_csv('../tesis_speechRate/src/pre_processing/tables/tables/TIMIT_df_by_sample_test.csv')

#%%
# READ FEATURES 
phonogram_features_TRAIN = pd.read_csv('../tesis_speechRate/src/processing/data_phonograms/data_features/Train/features_train.csv')
phonogram_features_TEST = pd.read_csv('../tesis_speechRate/src/processing/data_phonograms/data_features/Test/features_test.csv')

#%%
phonogram_features_TRAIN.set_index('sample_id', inplace=True)
phonogram_features_TEST.set_index('sample_id', inplace=True)


# Merge the phonogram features with the sample_phone dataframes
sample_phone_train.set_index('sample_id', inplace=True)
sample_phone_test.set_index('sample_id', inplace=True)
#%%
df_X_TRAIN = pd.merge(phonogram_features_TRAIN, sample_phone_train, left_index=True, right_index=True)

#%% -------------------------- SPLITTING -------------------------------------
speaker_id = df_X_TRAIN['speaker_id'].unique()
n_speakers = len(speaker_id)

# 80% Train - 20% Val
n_train = round(0.8*n_speakers)
n_val = n_speakers - n_train
# Choose randomly
random.shuffle(speaker_id)
speaker_id_train = speaker_id[:n_train]
speaker_id_val = speaker_id[n_train:]
#%%
# Filter the speaker_id_train
df_TRAIN = df_X_TRAIN[df_X_TRAIN['speaker_id'].isin(speaker_id_train)]
df_VAL = df_X_TRAIN[df_X_TRAIN['speaker_id'].isin(speaker_id_val)]


# %% TRAIN SET
X_TRAIN  = df_TRAIN.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_TRAIN = df_TRAIN['mean_speed_wpau_v1']
X_VAL = df_VAL.drop(columns=['region_id', 'speaker_id', 'mean_speed_wpau_v1', 'mean_speed_wpau_v2', 'mean_speed_wopau_v1', 'mean_speed_wopau_v2'])
y_VAL = df_VAL['mean_speed_wpau_v1']
# %% SPEAKERS DISTRIBUTION

# Speaker metrics
# Mean speed by speaker

mean_speaker = df_X_TRAIN.groupby('speaker_id')['mean_speed_wpau_v1'].mean()
# Standard deviation of the speed by speaker
std_speaker = df_X_TRAIN.groupby('speaker_id')['mean_speed_wpau_v1'].std()
#%%
# Merge
speaker_metrics = pd.concat([mean_speaker, std_speaker], axis=1)
speaker_metrics.columns = ['mean_speed_wpau_v1', 'std_speed_wpau_v1']
#%%
# The 3 speakers with the highest mean speed
fasters = speaker_metrics.nlargest(3, 'mean_speed_wpau_v1')
# The 3 speakers with the lowest mean speed
slowers =speaker_metrics.nsmallest(3, 'mean_speed_wpau_v1')

#%%

# Boxplot of the fasters and slowers speakers in the same plot 2 colors, one for fasters and one for slowers


# Filter the data
fasters_data = df_X_TRAIN[df_X_TRAIN['speaker_id'].isin(fasters.index)]
slowers_data = df_X_TRAIN[df_X_TRAIN['speaker_id'].isin(slowers.index)]

mean_speed = df_X_TRAIN['mean_speed_wpau_v1']

# I need 3 boxplots: 1 of the fasters, 1 of the slowers and 1 using the information of all the data

# array of each speaker


#%% array of each speaker

l_fasters = []
l_slowers = []
for i in fasters.index:
     l_fasters.append(df_X_TRAIN[df_X_TRAIN['speaker_id'] == i]['mean_speed_wpau_v1'])
for i in slowers.index:
     l_slowers.append(df_X_TRAIN[df_X_TRAIN['speaker_id'] == i]['mean_speed_wpau_v1'])

# In opposite order
l_slowers = l_slowers[::-1]


data = l_fasters + [mean_speed] + l_slowers


# Sample data

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns  # For better visual appeal

# Setting a style
sns.set(style="whitegrid")

# Sample data for demonstration

# Concatenating all data

# Define speaker categories
fasters_id = list(fasters.index)
slowers_id = list(slowers.index)
# Slowers in oposite order
slowers_id = slowers_id[::-1]

# Create the boxplot
fig, ax = plt.subplots(figsize=(10, 6))
box = plt.boxplot(data, patch_artist=True)

# Colors for each boxplot: red for fasters, green for 'all', blue for slowers
colors = ['r', 'r', 'r', 'g', 'b', 'b', 'b']

# Setting colors for each box
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# Enhancing the median line visibility
for median in box['medians']:
    median.set_color('k')
    median.set_linewidth(2)  # Make median lines thicker

# Adding labels and title
plt.ylabel('Velocidad Media', fontsize=14, fontweight='bold')
plt.xlabel('Hablantes (ID)', fontsize=14, fontweight='bold')
plt.title('Analisis de la velocidad del habla por hablante', fontsize=16, fontweight='bold')

# Adding custom x-tick labels
plt.xticks([1+0.15, 2+0.15, 3+0.15, 4+0.15, 5+0.15, 6+0.15, 7+0.15], fasters_id + ['Todos'] + slowers_id, rotation=0, ha='right', fontsize=12)

# Adding grid for better readability
plt.grid(True, linestyle='--', which='major', color='grey', alpha=0.5)

# Add a legend with custom handles
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor='r', label='Rápidos'),
                   Patch(facecolor='g', label='Todos'),
                   Patch(facecolor='b', label='Lentos')]
ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

# Show the plot
plt.tight_layout()  # Adjust the layout to make room for the rotated x-tick labels
plt.savefig('boxplot_speakers.png')

plt.show()



#%% The same boxplot but adding the mean speed of all the speakers (this box bigger than the others)


# %% PCA
from sklearn.decomposition import PCA

# linear model
from sklearn.linear_model import LinearRegression

A = ['all_mean_phonogram', 
     'all_mean_delta', 
     'all_mean_d_delta', 
     'all_std_phonogram',
     'all_mean_abs_phonogram',
     'all_mean_abs_delta',
     'all_mean_abs_d_delta'] 



#%% ALl PHONES FEATURES
# Means
B_mean = ['mean_phone_' + str(i) for i in range(2, 40)]
B_mean_delta = ['mean_delta_phone_' + str(i) for i in range(2, 40)]
B_mean_d_delta = ['mean_d_delta_phone_' + str(i) for i in range(2, 40)]

# Stds
B_std = ['std_phone_' + str(i) for i in range(2, 40)]
B_std_delta = ['std_delta_phone_' + str(i) for i in range(2, 40)]   
B_std_d_delta = ['std_d_delta_phone_' + str(i) for i in range(2, 40)]

# Abs
B_mean_abs = ['mean_abs_phone_' + str(i) for i in range(2, 40)]
B_mean_abs_delta = ['mean_abs_delta_phone_' + str(i) for i in range(2, 40)]
B_mean_abs_d_delta = ['mean_abs_d_delta_phone_' + str(i) for i in range(2, 40)]


B_softmax = ['feature_softmax_' + str(i) for i in range(2, 40)]
B_mean_softmax = ['mean_feature_softmax_' + str(i) for i in range(2, 40)]



B = B_mean + B_std + B_mean_delta + B_std_delta + B_mean_d_delta + B_std_d_delta 

B_abs = B_mean_abs + B_mean_abs_delta + B_mean_abs_d_delta

C = B_mean_softmax

D = ['greedy_feature']

BC = B + C

B_ABS_C = B_abs + C




#%% PCA
features = A+B+C+D+BC+B_ABS_C

X = X_TRAIN[B]
y = y_TRAIN
X_val = X_VAL[B]
y_val = y_VAL


#%% We want to predict y using the features

# linear model
from sklearn.linear_model import LinearRegression

# Create the model

model = LinearRegression(positive=True)

# Fit the model

model.fit(X, y)

# Predict the values

y_pred = model.predict(X_val)




#%%
# Evaluate the model
mse = mean_squared_error(y_val, y_pred)
print('MSE:', mse)

correlation = np.corrcoef(y_val, y_pred)[0, 1]

R2 = model.score(X_val, y_val)   
#%%
# Plot the prediction color the speed
sns.scatterplot(x=y_val, y=y_pred, hue=y_val, palette='coolwarm')


plt.xlabel('Velocidad Real')
plt.ylabel('Predicción')
plt.title('Prediccion con Atributo B')
# Add correlation in the legend
plt.legend(title='Correlation = ' + str(round(correlation,2)), loc='upper left')

plt.show()

#%% correlation



# %%
