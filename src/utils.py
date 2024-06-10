#%% Importing libraries
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import IPython.display as ipd
import numpy as np
import statsmodels.api as sm
import random
from datasets import load_dataset
import librosa
import torch 
import time
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
import seaborn as sns
from src.pre_processing.metrics.metrics import m1

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
    plt.ylabel('Speed')
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

    # ## NOTA: tiene que devolver 2 valores, el mean_speed y el mean_speed sin los silencios

    # v1
    amount_phones = x.loc[~x["utterance"].isin(silence),:].shape[0]
    amount_silences = x.loc[x["utterance"].isin(silence),:].shape[0]

    duration_wpau = (x['start'].iloc[-1] - x['start'].iloc[1]) / 16000
    
    duration_wopau = x.loc[~x["utterance"].isin(silence),:]['duration_s'].sum()

    avg_speed_wopau_v1 = amount_phones/duration_wopau
    avg_speed_wpau_v1 = amount_phones/duration_wpau

    # V2
    avg_dur_wopau = x.loc[~x["utterance"].isin(silence),:]["duration_s"].mean()
    avg_speed_wopau_v2 = 1/avg_dur_wopau

    #
    dur_wopau = x.loc[~x["utterance"].isin(silence)]["duration_s"].sum() 
    avg_dur_wpau = dur_wopau/(len(x)-2) # -2 because the first and last are not phones 
    avg_speed_wpau_v2 = 1/avg_dur_wpau

    return duration_wopau, avg_speed_wopau_v1, duration_wpau, avg_speed_wpau_v1, amount_phones, amount_silences, avg_speed_wopau_v2, avg_speed_wpau_v2

def duration_wpau(x):
    return mean_speed(x)[0]

def duration_wopau(x):
    return mean_speed(x)[2]

def avg_speed_wopau_v1(x):    
    return mean_speed(x)[1]

def avg_speed_wpau_v1(x):
    return mean_speed(x)[3]

def amount_phones(x):
    return mean_speed(x)[4]

def amount_silences(x):
    return mean_speed(x)[5]

def avg_speed_wopau_v2(x):
    return mean_speed(x)[6]

def avg_speed_wpau_v2(x):
    return mean_speed(x)[7]

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

# M3: Metric3 = |P(a)|/T(a) where P(a) is the amount of phones in the sample a and T(a) is the duration of the sample a.
# M2: Metric2 = 1/t(p_i) where t(p_i) is the duration of the phone p_i.


def TIMIT_df_by_sample_phones(df_by_record_of_phones):
    TIMIT_df_samples = pd.DataFrame()
    TIMIT_df_samples["duration_wpau"] = df_by_record_of_phones.groupby("sample_id").apply(duration)
    TIMIT_df_samples["duration_wopau"] = df_by_record_of_phones.groupby("sample_id").apply(duration_wopau)
    TIMIT_df_samples["mean_speed_wpau_v1"] = df_by_record_of_phones.groupby("sample_id").apply(avg_speed_wpau_v1)
    TIMIT_df_samples["mean_speed_wopau_v1"] = df_by_record_of_phones.groupby("sample_id").apply(avg_speed_wopau_v1)
    TIMIT_df_samples["mean_speed_wpau_v2"] = df_by_record_of_phones.groupby("sample_id").apply(avg_speed_wpau_v2)
    TIMIT_df_samples["mean_speed_wopau_v2"] = df_by_record_of_phones.groupby("sample_id").apply(avg_speed_wopau_v2)
    TIMIT_df_samples["amount_phones"] = df_by_record_of_phones.groupby("sample_id").apply(amount_phones)
    TIMIT_df_samples["amount_silences"] = df_by_record_of_phones.groupby("sample_id").apply(amount_silences)
    return TIMIT_df_samples
# %% 

#%% --------------------- FUNCTIONS ------------------------------------------
def get_phonograms(TIMIT_set, model, n_samples = 10,  train=True):
    t0 = time.time()
    phonograms = []
    

    print('--------------GETTING PHONOGRAMS-----------------')
    for i in range(n_samples):
        sample = TIMIT_set[i]

        sample_id = get_sample_ID(sample['dialect_region'], sample['speaker_id'], sample['id'])
        
        audio_data = sample['audio']['array']
        x = torch.tensor(np.array([audio_data]).astype(np.float32))
        with torch.no_grad():
            y = model(x).logits
        y = y.numpy()[0].T

        # Save each phonogram as a matrix the ohonograms are numpy.ndarray
        y_df = pd.DataFrame(y)
        if train:
            y_df.to_csv('../tesis_speechRate/src/processing/data_phonograms/CHARSIU/Train/' + sample_id + '.csv', index=False)
        else:
            y_df.to_csv('../tesis_speechRate/src/processing/data_phonograms/CHARSIU/Test/' + sample_id + '.csv', index=False)
        
        
       # phonograms.append([sample_id,y]) 
        if i % 10 == 0:

            print('SAMPLE ', i, ' OF ', n_samples)

    t1 = time.time()
    print('-------------------------------------------------')
    print('Time to get phonograms: ', t1-t0)

# Dataframes with the array of the samples (array_id, array)

def get_sample_ID(dialect_region, speaker_id, id):
    return dialect_region + '_' + speaker_id + '_' + id

def get_sample_IDs(TIMIT_set, n_samples = 10):
    sample_id = []
    for i in range(n_samples):
        sample = TIMIT_set[i]
        sample_id.append(get_sample_ID(sample['dialect_region'], sample['speaker_id'], sample['id']))
    return sample_id

def get_dialectRegion_and_speacker_ID(sample_ID):
    # Sample_ID is an string, for example: 'DR1_CJF0_SA1'
    # Until the first '_' is Dialect region
    # Until the second '_' is the speaker ID
    parts = sample_ID.split('_')
    DR_ID = parts[0]
    speaker_ID = parts[1]
    return DR_ID, speaker_ID


def get_phonograms_from_csv(sample_ID, train=True):
    if train:
        phonogram = pd.read_csv('../tesis_speechRate/src/processing/data_phonograms/CHARSIU/Train/'+sample_ID+'.csv')
    else: 
        phonogram = pd.read_csv('../tesis_speechRate/src/processing/data_phonograms/CHARSIU/Test/'+sample_ID+'.csv')
    return phonogram.to_numpy()

#  Phonogram features
def phonogram_to_features(sample_ID, train=True, begin=0, end=0):


    phonogram = get_phonograms_from_csv(sample_ID, train)
    if end == 0:
        end = phonogram.shape[1]
    phonogram = phonogram[:, begin:end]
    delta = librosa.feature.delta(phonogram)
    d_delta = librosa.feature.delta(phonogram, order=2)
    
    DR_ID, speaker_ID = get_dialectRegion_and_speacker_ID(sample_ID=sample_ID)

    dic = {'sample_id': sample_ID}
    dic['region_id'] =  DR_ID
    dic['speaker_id'] = speaker_ID
    
    # TODO: Vectorizar al estilo numpy

    # Mean each phonogram
    means = np.mean(phonogram, axis=1)
    mean_dic = {f'mean_phone_{i+1}': mean for i, mean in enumerate(means)}
    dic.update(mean_dic)

    # STD each phonogram
    stds = np.std(phonogram, axis=1)
    std_dic = {f'std_phone_{i+1}': std for i, std in enumerate(stds)}
    dic.update(std_dic)
    
    # STD delta each phonogram
    mean_delta = np.mean(delta, axis=1)
    mean_delta_dic = {f'mean_delta_phone_{i+1}': mean for i, mean in enumerate(mean_delta)}
    dic.update(mean_delta_dic)

    # STD delta each phonogram
    std_delta = np.std(delta, axis=1)
    std_delta_dic = {f'std_delta_phone_{i+1}': std for i, std in enumerate(std_delta)}
    dic.update(std_delta_dic)

    # STD delta each phonogram
    mean_d_delta = np.mean(d_delta, axis=1)
    mean_d_delta_dic = {f'mean_d_delta_phone_{i+1}': mean for i, mean in enumerate(mean_d_delta)}
    dic.update(mean_d_delta_dic)

    # STD delta each phonogram
    std_d_delta = np.std(d_delta, axis=1)
    std_d_delta_dic = {f'std_d_delta_phone_{i+1}': std for i, std in enumerate(std_d_delta)}
    dic.update(std_d_delta_dic)

    # Mean absolute value of each phonogram
    abs_mean_phonogram = np.mean(np.abs(phonogram), axis=1)
    abs_mean_phonogram_dic = {f'mean_abs_phone_{i+1}': mean for i, mean in enumerate(abs_mean_phonogram)}
    dic.update(abs_mean_phonogram_dic)

    # Mean absolute value of each delta
    mean_abs_delta = np.mean(np.abs(delta), axis=1)
    mean_abs_delta_dic = {f'mean_abs_delta_phone_{i+1}': mean for i, mean in enumerate(mean_abs_delta)}
    dic.update(mean_abs_delta_dic)

    # Mean absolute value of each delta
    mean_abs_d_delta = np.mean(np.abs(d_delta), axis=1)
    mean_abs_d_delta_dic = {f'mean_abs_d_delta_phone_{i+1}': mean for i, mean in enumerate(mean_abs_d_delta)}
    dic.update(mean_abs_d_delta_dic)

    # Add the feature of softmax
    feature_softmax = how_many_probables_phones(phonogram)[0]
    dic_feature_softmax = {f'feature_softmax_{i+1}': feature for i, feature in enumerate(feature_softmax)}
    dic.update(dic_feature_softmax)

    # Add the mean feature of softmax
    mean_feature_softmax = how_many_probables_phones(phonogram)[1]
    dic_mean_feature_softmax = {f'mean_feature_softmax_{i+1}': feature for i, feature in enumerate(mean_feature_softmax)}
    dic.update(dic_mean_feature_softmax)    


    # Means realated of all the phonogram
    dic['all_mean_phonogram'] = np.mean(phonogram)
    dic['all_mean_delta'] = np.mean(delta)
    dic['all_mean_d_delta'] = np.mean(d_delta)
    dic['all_std_phonogram'] = np.std(phonogram)
    dic['all_mean_abs_phonogram'] = np.mean(np.abs(phonogram))
    dic['all_mean_abs_delta'] = np.mean(np.abs(delta))
    dic['all_mean_abs_d_delta'] = np.mean(np.abs(d_delta))
    
    dic['greedy_feature'] = greedy_feature(phonogram)[0]
    
    
    features = pd.DataFrame(dic, index=[0])
    # Save the features as a csv file
    return features


def softmax_phonogram(phonogram):
    # dataframe to torch
    phonogram = torch.tensor(phonogram.astype(np.float32))
    phonogram_softmax = torch.softmax(phonogram, dim=0)
    #to numpy
    phonogram_softmax = phonogram_softmax.numpy()

    return phonogram_softmax

def greedy_feature(phonogram, t=0):
    T = phonogram.shape[1]
    
    if t == 0:
        t = T

    s = np.zeros(t)
    how_many_until_i = 0

    argmax = -1
    for i in range(t):
        arg_max_new = np.argmax(phonogram[:,i])
        is_a_silence = arg_max_new == 0
        is_a_new_phone = arg_max_new != argmax 

        if (is_a_new_phone) and (not is_a_silence):
            argmax = arg_max_new
            how_many_until_i += 1
        s[i] = how_many_until_i
    
    mean_most_probable_phones = how_many_until_i/t

    return mean_most_probable_phones,s
        
#def mean_phones_features(phonogram):
#    how_many_phones_argMax = how_many_phones_since_t(phonogram)[0]
#    how_many_probables_phone = how_many_probables_phones(phonogram)[0]
#    T = phonogram.shape[1]/100 # 100 is the number of frames per second
#    return how_many_phones_argMax/T, how_many_probables_phone/T

def how_many_probables_phones(phonogram, t=0):
    number_of_phones = 42
    T = phonogram.shape[1]
    if t == 0:
        t = T
    phonogram_softmax = softmax_phonogram(phonogram)
    phonogram_softmax = phonogram_softmax > 0.5
    
    d_phonogram_softmax = np.diff(phonogram_softmax, axis=1)

    res  = np.zeros(number_of_phones)
    for i in range(number_of_phones):
        res[i] = np.sum(d_phonogram_softmax[i,:])

    

    return res, res/t



#%% =================================== GENERATE THE INSTANTANOUS SPEED ===================================


def get_real_speed_CHARSIU_TEXTLESS(sample_ids, data, train=True, step_size=1/60*16000, window_size=1/4*16000, with_pause=False):
    """
    This function returns the real speed of the samples in the sample_ids list.
    """
    print('PROCESSING ' + 'TRAIN' if train else 'TEST')
    N = len(sample_ids)
    
    t0 = time.time()

    for i in range(N):

        if i % 100 == 0:
            print('Processing sample ' + str(i) + ' of ' + str(N))
        
        speed_df = pd.DataFrame()
        sample_id = sample_ids[i]
        sample_features = data[data['sample_id'] == sample_id] if train else data[data['sample_id'] == sample_id]
        
        #reset index
        sample_features = sample_features.reset_index(drop=True)
        
        x, y = m1(sample_features, step_size=step_size, window_size=window_size, with_pau=with_pause)
        speed_df['x'] = x
        speed_df['y'] = y


    tf = time.time()

    print('DONE. Time:' + str(tf-t0) + 's')


def r2_experiment(df_train, df_val, df_test, y_train, y_val, y_test, features, N=10):
    MSE_features = np.zeros(len(features))
    scores = np.zeros(len(features))
    test_scores = np.zeros(len(features))
    for j in range(N):
        for i in range(len(features)):
            print('Features:', features[i])
            X_TRAIN_fi = df_train[features[i]]
            X_VAL_fi = df_val[features[i]]
            
            # Regression
            positive=True
            model = linear_model.LinearRegression(positive=positive)
            model.fit(X_TRAIN_fi, y_train)
            y_pred = model.predict(X_VAL_fi)
            MSE_features[i] += mean_squared_error(y_val, y_pred)
            scores[i] += model.score(X_VAL_fi, y_val)

            # Report scores in test

            X_TEST_fi = df_test[features[i]]
            y_pred = model.predict(X_TEST_fi)
            test_score = model.score(X_TEST_fi, y_test)
            test_scores[i] += test_score

    mean_score_features = scores/N
    mean_MSE_features = MSE_features/N
    mean_test_score = test_scores/N
    return mean_score_features, mean_MSE_features, model, mean_test_score