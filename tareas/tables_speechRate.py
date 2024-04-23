#%%
import pandas as pd
import utils as ut
#%%  Load the Dataframes of each record

TIMIT_df_by_record = ut.TIMIT_df_by_record()   
TIMIT_df_by_record.build_phone_test(TIMIT_test)
TIMIT_df_by_record.build_phone_train(TIMIT_train)
TIMIT_df_by_record.build_word_test(TIMIT_test)
TIMIT_df_by_record.build_word_train(TIMIT_train)

# %% Load the Dataframes of each sample
TIMIT_df_by_sample_train = ut.TIMIT_df_by_sample(TIMIT_df_by_record.phone_train)
TIMIT_df_by_sample_test = ut.TIMIT_df_by_sample(TIMIT_df_by_record.phone_test)


# %%
TIMIT_df_by_sample_word_test = ut.TIMIT_df_by_sample(TIMIT_df_by_record.word_test)

# %%
