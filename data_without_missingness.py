import pandas as pd
from datasets import datasets
import matplotlib.pyplot as plt
from sdv import load_demo
from sdv.tabular import CTGAN
from table_evaluator import TableEvaluator

all_datasets = ["mushroom", "credit", "letter", "bank", "news"]
# all_datasets = ["letter"]
all_missingness = [10,30,50]
# all_missingness = [10]

for data_name in all_datasets:
    for missingness in all_missingness:
        filename_missing_data = 'ohe_preprocessed_data/one_hot_test_data/one_hot_{}_test_{}.csv'.format(data_name, missingness)
        missing_data_wo_target = pd.read_csv(filename_missing_data)

        # Remove out rows with missing values and save to a new DataFrame
        data_no_nans = missing_data_wo_target.dropna()
        
        filename = 'ohe_preprocessed_data/one_hot_test_data_missing_rows_deleted/one_hot_{}_test_{}.csv'.format(data_name, missingness)
        data_no_nans.to_csv(filename, index=False)
