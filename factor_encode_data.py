
import pandas as pd
import numpy as np
from datasets import datasets

''' Description

xx

'''
# Determine dataset, missingness and mode (test/train)
all_datasets = ["mushroom", "news", "credit", "letter", "bank"]
all_missingness = [10, 30, 50, 70]

for dataset in all_datasets:
    for missingness in all_missingness:
        # Find categorical and target column
        cat_cols = datasets[dataset]["cat_cols"]
        target_col = datasets[dataset]["target"]

        # Concatenate complete datasets
        filename_train_complete = 'train_data/{}_train.csv'.format(dataset)
        train_data_complete = pd.read_csv(filename_train_complete)

        filename_test_complete = 'test_data/{}_test.csv'.format(dataset)
        test_data_complete = pd.read_csv(filename_test_complete)

        full_data_complete = pd.concat([train_data_complete, test_data_complete], axis=0)
        target_col_full_data_complete = full_data_complete[target_col]
        full_data_complete = full_data_complete.drop(target_col, axis=1) # Full data without missingness

        # Concatenate datasets with missingness
        filename_train_x = 'train_data/{}_train_{}.csv'.format(dataset, missingness)
        train_data_x = pd.read_csv(filename_train_x)

        filename_test_x = 'test_data/{}_test_{}.csv'.format(dataset, missingness)
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
            mapping = dict(zip(unique_values, range(len(unique_values))))
            df_full_data_complete[col_encoded_name] = df_full_data_complete[col].replace(mapping)
            df_full_data_complete[col_encoded_name] = df_full_data_complete[col_encoded_name].replace({np.nan: np.nan})

            unique_values = df_full_data_x[col].dropna().unique()  # exclude np.nan values
            mapping = dict(zip(unique_values, range(len(unique_values))))
            df_full_data_x[col_encoded_name] = df_full_data_x[col].replace(mapping)
            df_full_data_x[col_encoded_name] = df_full_data_x[col_encoded_name].replace({np.nan: np.nan})
                            
            # Remove the original categorical columns from the new dataframe
            df_full_data_x.drop(col, axis=1, inplace=True)
            df_full_data_complete.drop(col, axis=1, inplace=True)

        # Add back the target column
        df_full_data_x[target_col] = target_col_full_data_x
        df_full_data_complete[target_col] = target_col_full_data_complete

        # Split back into training and test
        train_data_complete, test_data_complete = np.vsplit(df_full_data_complete, [len(train_data_complete)])
        train_data_x, test_data_x = np.vsplit(df_full_data_x, [len(train_data_x)])

        # Save to CSV
        filename_train_complete = 'factor_train_data/factor_encode_{}_train.csv'.format(dataset)
        train_data_complete.to_csv(filename_train_complete, index=False)
        filename_test_complete = 'factor_test_data/factor_encode_{}_test.csv'.format(dataset)
        test_data_complete.to_csv(filename_test_complete, index=False)

        filename_train_x = 'factor_train_data/factor_encode_{}_train_{}.csv'.format(dataset, missingness)
        train_data_x.to_csv(filename_train_x, index=False)
        filename_test_x = 'factor_test_data/factor_encode_{}_test_{}.csv'.format(dataset, missingness)
        test_data_x.to_csv(filename_test_x, index=False)

        print(test_data_x.shape)
        print(train_data_x.shape)

        print(test_data_complete.shape)
        print(train_data_complete.shape)
