
import pandas as pd
import numpy as np
from datasets import datasets
from data_loader import data_loader_ohe_wo_target

''' Description

xx

'''
# Determine dataset, missingness and mode (test/train)
all_datasets = ["mushroom", "news", "credit", "letter", "bank"]
all_missingness = [10,30,50]

for data_name in all_datasets:
    for missingness in all_missingness:

        # Define miss rate
        miss_rate = missingness/100

        # Load OHE data
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target(data_name, miss_rate)
        data_to_encode = [train_data_x, train_miss_data_x, test_data_x, test_miss_data_x] 

        # Find total number of columns and number of numerical columns 
        _, dim_train = train_data_x.shape
        N_num_cols = len(datasets[data_name]["num_cols"])

        # Find indicies for categorical columns 
        cat_index = list(range(N_num_cols,dim_train))

        # Save encoded data
        data_encoded = []

        if len(cat_index) == 0:
            train_enc_data_x, train_miss_enc_data_x, test_enc_data_x, test_miss_enc_data_x = data_to_encode
        else:
            for df in data_to_encode:
                columns_to_rename = df.columns[cat_index]
                new_column_names = {col: col+'_encoded' for col in columns_to_rename}
                df_encoded = df.rename(columns=new_column_names)
                data_encoded.append(df_encoded)

            train_enc_data_x, train_miss_enc_data_x, test_enc_data_x, test_miss_enc_data_x = data_encoded

        # Save to CSV
        filename_train_complete = 'one_hot_train_data_wo_target_for_mice/one_hot_for_mice_{}_train.csv'.format(data_name)
        train_enc_data_x.to_csv(filename_train_complete, index=False)
        filename_test_complete = 'one_hot_test_data_wo_target_for_mice/one_hot_for_mice_{}_test.csv'.format(data_name)
        test_enc_data_x.to_csv(filename_test_complete, index=False)

        filename_train_x = 'one_hot_train_data_wo_target_for_mice/one_hot_for_mice_{}_train_{}.csv'.format(data_name, missingness)
        train_miss_enc_data_x.to_csv(filename_train_x, index=False)
        filename_test_x = 'one_hot_test_data_wo_target_for_mice/one_hot_for_mice_{}_test_{}.csv'.format(data_name, missingness)
        test_miss_enc_data_x.to_csv(filename_test_x, index=False)

        print(train_enc_data_x.shape)
        print(test_enc_data_x.shape)

        print(train_miss_enc_data_x.shape)
        print(test_miss_enc_data_x.shape)
