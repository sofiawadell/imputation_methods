
import pandas as pd
import numpy as np
from datasets import datasets

''' Description

Performs factor encoding for the categorical columns in all train data sets that have been increased using CTGAN, so that the subcategories in each column are represented by a number. Adds "_encoded" to all the categorical column names. 
Subcategories within each categorical column are factor encoded in alphabetical order, so the first subcategory according to alphabetical order is represented by a 0, the second one by a 1 and so on. 
Both full data sets and data sets with missingness are encoded and saved. Missing values are represented as np.nan.

'''
# Determine dataset, missingness 
#all_datasets = ["mushroom", "credit", "letter", "bank"]
all_datasets = ["news"]
all_missingness = [10]

for dataset in all_datasets:
    for missingness in all_missingness:

        # Find categorical and target column
        cat_cols = datasets[dataset]["cat_cols"]
        target_col = datasets[dataset]["target"]

        # Concatenate complete datasets
        filename_train_complete = 'train_test_split_data/train_data/{}_train.csv'.format(dataset)
        train_data_complete = pd.read_csv(filename_train_complete)

        filename_test_complete = 'train_test_split_data/test_data/{}_test.csv'.format(dataset)
        test_data_complete = pd.read_csv(filename_test_complete)

        full_data_complete = pd.concat([train_data_complete, test_data_complete], axis=0)
        target_col_full_data_complete = full_data_complete[target_col]
        full_data_complete = full_data_complete.drop(target_col, axis=1) # Full data without missingness without target column

        # Load CTGAN train datasets with missingness
        filename_train_50_x = 'train_test_split_data/train_data_wo_target_extra_50/{}_train_{}_extra_50.csv'.format(dataset, missingness)
        train_data_50_x = pd.read_csv(filename_train_50_x)

        filename_train_100_x = 'train_test_split_data/train_data_wo_target_extra_100/{}_train_{}_extra_100.csv'.format(dataset, missingness)
        train_data_100_x = pd.read_csv(filename_train_100_x)

        # Create copy of dataframes
        df_full_data_complete = full_data_complete.copy()
        df_train_data_100_x = train_data_100_x.copy()
        df_train_data_50_x = train_data_50_x.copy()

        # Loop through each categorical column and apply factor encoding
        for col in cat_cols:

            col_encoded_name = col + '_encoded'
            unique_values = df_full_data_complete[col].dropna().unique()  # exclude np.nan values
            unique_values_sorted = sorted(unique_values)
            mapping = dict(zip(unique_values_sorted, range(len(unique_values_sorted))))

            df_train_data_50_x[col_encoded_name] = df_train_data_50_x[col].replace(mapping)
            df_train_data_50_x[col_encoded_name] = df_train_data_50_x[col_encoded_name].replace({np.nan: np.nan})

            df_train_data_100_x[col_encoded_name] = df_train_data_100_x[col].replace(mapping)
            df_train_data_100_x[col_encoded_name] = df_train_data_100_x[col_encoded_name].replace({np.nan: np.nan})
                            
            # Remove the original categorical columns from the new dataframe
            df_train_data_50_x.drop(col, axis=1, inplace=True)
            df_train_data_100_x.drop(col, axis=1, inplace=True)

        ## Save data without target column 

        # Save to CSV
        filename_train_extra_50 = 'factor_preprocessed_data/factor_train_data_wo_target_extra_50/factor_encode_{}_train_{}_extra_50.csv'.format(dataset, missingness)
        df_train_data_50_x.to_csv(filename_train_extra_50, index=False)

        filename_train_extra_100 = 'factor_preprocessed_data/factor_train_data_wo_target_extra_100/factor_encode_{}_train_{}_extra_100.csv'.format(dataset, missingness)
        df_train_data_100_x.to_csv(filename_train_extra_100, index=False)



        print(df_train_data_50_x.shape)
        print(df_train_data_100_x.shape)

        print(df_full_data_complete.shape)
