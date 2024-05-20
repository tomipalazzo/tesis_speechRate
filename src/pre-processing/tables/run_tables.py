#%%
import pandas as pd
import src.utils as ut
from datasets import load_dataset


#%% 
TIMIT = load_dataset('timit_asr', data_dir='/home/tomi/Documents/tesis_speechRate/timit')
TIMIT_train = TIMIT['train']
TIMIT_test = TIMIT['test']

#%%  Load the Dataframes of each record

TIMIT_df_by_record = ut.TIMIT_df_by_record()   
TIMIT_df_by_record.build_phone_test(TIMIT_test)
TIMIT_df_by_record.build_phone_train(TIMIT_train)
TIMIT_df_by_record.build_word_test(TIMIT_test)
TIMIT_df_by_record.build_word_train(TIMIT_train)

# %% Load the Dataframes of each sample
TIMIT_df_by_sample_train = ut.TIMIT_df_by_sample_phones(TIMIT_df_by_record.phone_train)
TIMIT_df_by_sample_test = ut.TIMIT_df_by_sample_phones(TIMIT_df_by_record.phone_test)


# %%
TIMIT_df_by_sample_word_test = ut.TIMIT_df_by_sample_phones(TIMIT_df_by_record.word_test)

# %% Save the dataframes
TIMIT_df_by_record.phone_test.to_csv('../tesis_speechRate/src/pre-processing/tables/tables/TIMIT_df_by_record_phone_test.csv', index=True)
TIMIT_df_by_record.phone_train.to_csv('../tesis_speechRate/src/pre-processing/tables/tables/TIMIT_df_by_record_phone_train.csv', index=True)
TIMIT_df_by_record.word_test.to_csv('../tesis_speechRate/src/pre-processing/tables/tables/TIMIT_df_by_record_word_test.csv', index=True)
TIMIT_df_by_record.word_train.to_csv('../tesis_speechRate/src/pre-processing/tables/tables/TIMIT_df_by_record_word_train.csv', index=True)
TIMIT_df_by_sample_train.to_csv('../tesis_speechRate/src/pre-processing/tables/tables/TIMIT_df_by_sample_train.csv', index=True)
TIMIT_df_by_sample_test.to_csv('../tesis_speechRate/src/pre-processing/tables/tables/TIMIT_df_by_sample_test.csv', index=True)





# %%
