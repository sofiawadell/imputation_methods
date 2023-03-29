
import pandas as pd
import numpy as np
from datasets import datasets

''' Description

Performs factor encoding for the categorical columns in all data sets, so that the subcategories in each column are represented by a number. Adds "_encoded" to all the categorical column names. 
Subcategories within each categorical column are factor encoded in alphabetical order, so the first subcategory according to alphabetical order is represented by a 0, the second one by a 1 and so on. 
Both full data sets and data sets with missingness are encoded and saved. Missing values are represented as np.nan.

'''
# Determine dataset, missingness 
all_datasets = ["mushroom", "news", "credit", "letter", "bank"]
all_missingness = [10, 30, 50]

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
        full_data_complete = full_data_complete.drop(target_col, axis=1) # Full data without missingness

        # Concatenate datasets with missingness
        filename_train_x = 'train_test_split_data/train_data/{}_train_{}.csv'.format(dataset, missingness)
        train_data_x = pd.read_csv(filename_train_x)

        filename_test_x = 'train_test_split_data/test_data/{}_test_{}.csv'.format(dataset, missingness)
        test_data_x = pd.read_csv(filename_test_x)

        full_data_x = pd.concat([train_data_x, test_data_x], axis=0)
        target_col_full_data_x = full_data_x[target_col]
        full_data_x = full_data_x.drop(target_col, axis=1) # Full data with missingness without target column

        # Create copy of dataframes
        df_full_data_complete = full_data_complete.copy()
        df_full_data_x = full_data_x.copy()

        # Loop through each categorical column and apply factor encoding
        for col in cat_cols:

            col_encoded_name = col + '_encoded'
            unique_values = df_full_data_complete[col].dropna().unique()  # exclude np.nan values
            unique_values_sorted = sorted(unique_values)
            mapping = dict(zip(unique_values_sorted, range(len(unique_values_sorted))))

            df_full_data_complete[col_encoded_name] = df_full_data_complete[col].replace(mapping)

            df_full_data_x[col_encoded_name] = df_full_data_x[col].replace(mapping)
            df_full_data_x[col_encoded_name] = df_full_data_x[col_encoded_name].replace({np.nan: np.nan})
                            
            # Remove the original categorical columns from the new dataframe
            df_full_data_x.drop(col, axis=1, inplace=True)
            df_full_data_complete.drop(col, axis=1, inplace=True)

        ## Save data without target column 
        # Split back into training and test
        train_data_complete, test_data_complete = np.vsplit(df_full_data_complete, [len(train_data_complete)])
        train_data_x, test_data_x = np.vsplit(df_full_data_x, [len(train_data_x)])

        # Save to CSV
        filename_train_complete = 'factor_preprocessed_data/factor_train_data_wo_target/factor_encode_{}_train.csv'.format(dataset)
        train_data_complete.to_csv(filename_train_complete, index=False)
        filename_test_complete = 'factor_preprocessed_data/factor_test_data_wo_target/factor_encode_{}_test.csv'.format(dataset)
        test_data_complete.to_csv(filename_test_complete, index=False)

        filename_train_x = 'factor_preprocessed_data/factor_train_data_wo_target/factor_encode_{}_train_{}.csv'.format(dataset, missingness)
        train_data_x.to_csv(filename_train_x, index=False)
        filename_test_x = 'factor_preprocessed_data/factor_test_data_wo_target/factor_encode_{}_test_{}.csv'.format(dataset, missingness)
        test_data_x.to_csv(filename_test_x, index=False)

        ## Save data with target column 
        # Add back the target column
        df_full_data_x[target_col] = target_col_full_data_x
        df_full_data_complete[target_col] = target_col_full_data_complete

        # Split back into training and test
        train_data_complete, test_data_complete = np.vsplit(df_full_data_complete, [len(train_data_complete)])
        train_data_x, test_data_x = np.vsplit(df_full_data_x, [len(train_data_x)])

        # Save to CSV
        filename_train_complete = 'factor_preprocessed_data/factor_train_data/factor_encode_{}_train.csv'.format(dataset)
        train_data_complete.to_csv(filename_train_complete, index=False)
        filename_test_complete = 'factor_preprocessed_data/factor_test_data/factor_encode_{}_test.csv'.format(dataset)
        test_data_complete.to_csv(filename_test_complete, index=False)

        filename_train_x = 'factor_preprocessed_data/factor_train_data/factor_encode_{}_train_{}.csv'.format(dataset, missingness)
        train_data_x.to_csv(filename_train_x, index=False)
        filename_test_x = 'factor_preprocessed_data/factor_test_data/factor_encode_{}_test_{}.csv'.format(dataset, missingness)
        test_data_x.to_csv(filename_test_x, index=False)

        print(test_data_x.shape)
        print(train_data_x.shape)

        print(test_data_complete.shape)
        print(train_data_complete.shape)
