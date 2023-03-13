
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
        target_col = datasets[dataset]["target"]

        # Load full test data set
        filename_test_complete = 'factor_test_data/factor_encode_{}_test.csv'.format(dataset)
        test_data_complete = pd.read_csv(filename_test_complete)

        # Load test data sets with missingness
        filename_test_x = 'factor_test_data/factor_encode_{}_test_{}.csv'.format(dataset, missingness)
        test_data_x = pd.read_csv(filename_test_x)

        # Load full train data set
        filename_train_complete = 'factor_train_data/factor_encode_{}_train.csv'.format(dataset)
        train_data_complete = pd.read_csv(filename_train_complete)

        # Load test data sets with missingness
        filename_train_x = 'factor_train_data/factor_encode_{}_train_{}.csv'.format(dataset, missingness)
        train_data_x = pd.read_csv(filename_train_x)

        # Remove target column
        test_data_x_wo_target = test_data_x.drop(target_col, axis=1)
        test_data_complete_wo_target = test_data_complete.drop(target_col, axis=1)

        train_data_x_wo_target = train_data_x.drop(target_col, axis=1)
        train_data_complete_wo_target = train_data_complete.drop(target_col, axis=1)

        # Save to CSV
        filename_train_complete = 'factor_train_data_wo_target/factor_encode_{}_train.csv'.format(dataset)
        train_data_complete_wo_target.to_csv(filename_train_complete, index=False)
        filename_test_complete = 'factor_test_data_wo_target/factor_encode_{}_test.csv'.format(dataset)
        test_data_complete_wo_target.to_csv(filename_test_complete, index=False)

        filename_train_x = 'factor_train_data_wo_target/factor_encode_{}_train_{}.csv'.format(dataset, missingness)
        train_data_x_wo_target.to_csv(filename_train_x, index=False)
        filename_test_x = 'factor_test_data_wo_target/factor_encode_{}_test_{}.csv'.format(dataset, missingness)
        test_data_x_wo_target.to_csv(filename_test_x, index=False)

        print(test_data_x.shape)
        print(train_data_x.shape)

        print(test_data_complete.shape)
        print(train_data_complete.shape)
