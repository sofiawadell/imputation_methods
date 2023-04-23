import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np

from run_imputation_methods import run_mf
from missforest_impute import missforest_impute

from data_loader import data_loader_ohe_wo_target_ctgan50
from data_loader import data_loader_ohe_wo_target_ctgan100
from data_loader import data_loader_ohe_wo_target

from utils import normalize_numeric
from utils import renormalize_numeric
from utils import rmse_num_loss
from utils import rmse_cat_loss
from utils import m_rmse_loss
from utils import pfc
from utils import find_average_and_st_dev<
from utils import round_if_not_none

import time

''' Description

Imputes data set using the imputation methods MissForest, kNN or median/mode imputation and prints the result

'''

def run_missforest(data_name, miss_rate, ctgan, no_of_runs):
    
    results = []

    ########## Load data #######
    # Load OHE data without target column depending on ctgan variable
    if ctgan == "50":
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target_ctgan50(data_name, miss_rate) 
    elif ctgan == "100":
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target_ctgan100(data_name, miss_rate) 
    else:            
        train_data_x, train_miss_data_x, test_data_x, test_miss_data_x = data_loader_ohe_wo_target(data_name, miss_rate) 

    # Define mask matrix
    mask_train = pd.DataFrame(1-np.isnan(train_miss_data_x.values))
    mask_test = pd.DataFrame(1-np.isnan(test_miss_data_x.values)) 

    # Normalize the test data set using the norm_parameters from the training data with missingness
    train_miss_norm_data_x, norm_params_train_miss_data = normalize_numeric(train_miss_data_x, data_name)
    test_miss_norm_data_x, _ = normalize_numeric(test_miss_data_x, data_name, norm_params_train_miss_data)

    ## Part that should be done for no_of_runs
    for i in range(no_of_runs):
        # Start timer
        start_time = time.time()

        # Impute data using missForest
        test_imp_norm_data_x_np, train_imp_norm_data_x_np = missforest_impute(train_miss_norm_data_x, test_miss_norm_data_x, data_name)

        # End timer
        end_time = time.time()
        ex_time = end_time - start_time

        # Transform to pandas array to keep the column names 
        train_imp_norm_data_x = pd.DataFrame(train_imp_norm_data_x_np, columns=test_miss_data_x.columns) 
        test_imp_norm_data_x = pd.DataFrame(test_imp_norm_data_x_np, columns=test_miss_data_x.columns) 

        # Renormalize data sets
        train_imp_data_x = renormalize_numeric(train_imp_norm_data_x, norm_params_train_miss_data, data_name)
        test_imp_data_x = renormalize_numeric(test_imp_norm_data_x, norm_params_train_miss_data, data_name)

        # Save renormalized data 
        missingness = int(miss_rate*100)
        
        # Save imputed data based on ctgan variable 
        if ctgan == "50":
            filename_test_imp_ctgan50 = 'imputed_data/ctgan50/imputed_missforest_test_data/{}/imputed_missforest_{}_test_{}_ctgan50_{}.csv'.format(data_name, data_name, missingness, i)
            test_imp_data_x.to_csv(filename_test_imp_ctgan50, index=False)
        elif ctgan == "100":
            filename_test_imp_ctgan100 = 'imputed_data/ctgan100/imputed_missforest_test_data/{}/imputed_missforest_{}_test_{}_ctgan100_{}.csv'.format(data_name, data_name, missingness, i)
            test_imp_data_x.to_csv(filename_test_imp_ctgan100, index=False)
        else: 
            filename_test_imp = 'imputed_data/no_ctgan/imputed_missforest_test_data/{}/imputed_missforest_{}_test_{}_{}.csv'.format(data_name, data_name, missingness, i)
            test_imp_data_x.to_csv(filename_test_imp, index=False)

        # Normalize the imputed data set using the full data set 
        train_full_data_norm_x, norm_params_full_data_train = normalize_numeric(train_data_x, data_name)
        test_full_data_norm_x, _ = normalize_numeric(test_data_x, data_name, norm_params_full_data_train)
        test_imp_data_norm_x, _ = normalize_numeric(test_imp_data_x, data_name, norm_params_full_data_train) 

        # Calculate RMSE for numerical data
        rmse_num = rmse_num_loss(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)

        # Calculate RMSE for numerical and categorical data 
        rmse_cat = rmse_cat_loss(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)
        m_rmse = m_rmse_loss(rmse_num, rmse_cat)

        # Calculate PFC for categorical data 
        pfc_value = pfc(test_full_data_norm_x, test_imp_data_norm_x, mask_test, data_name)
  
        results.append({'run number': i, 'data': test_imp_data_x, 'scores':{'mRMSE': m_rmse, 'RMSE num': rmse_num, 'RMSE cat': rmse_cat, 'PFC': pfc_value, 'Execution time': ex_time}})

    average_m_rmse, st_dev_m_rmse = map(round_if_not_none, find_average_and_st_dev([x['scores']['mRMSE'] for x in results]))
    average_rmse_num, st_dev_rmse_num = map(round_if_not_none, find_average_and_st_dev([x['scores']['RMSE num'] for x in results]))
    average_rmse_cat, st_dev_rmse_cat = map(round_if_not_none, find_average_and_st_dev([x['scores']['RMSE cat'] for x in results]))
    average_pfc, st_dev_pfc = map(round_if_not_none, find_average_and_st_dev([x['scores']['PFC'] for x in results]))
    average_exec_time, st_dev_exec_time = map(round_if_not_none, find_average_and_st_dev([x['scores']['Execution time'] for x in results]))

    # Print the results
    print()
    print(f"Dataset: {data_name}, Miss_rate: {miss_rate}, Extra CTGAN data amount :{ctgan}")
    print(f"Average mRMSE: {average_m_rmse}, Standard deviation: {st_dev_m_rmse}")
    print(f"Average RMSE num: {average_rmse_num}, Standard deviation: {st_dev_rmse_num}")
    print(f"Average RMSE cat: {average_rmse_cat}, Standard deviation: {st_dev_rmse_cat}")
    print(f"Average PFC (%): {average_pfc}, Standard deviation: {st_dev_pfc}")
    print(f"Average execution time (sec): {average_exec_time}, Standard deviation: {st_dev_exec_time}")

    return average_m_rmse, st_dev_m_rmse, average_rmse_num, st_dev_rmse_num, average_rmse_cat, st_dev_rmse_cat, average_pfc, st_dev_pfc, average_exec_time, st_dev_exec_time


# Run main code for kNN, missForest and median/mode imputation
miss = [0.1, 0.3]
dat =["mushroom", "letter", "bank", "credit", "news"]
ctgan = ["50", "100"]
no_of_runs = 10

df_results = pd.DataFrame(columns=['Dataset', 'Missing%', 'Additional CTGAN data%', 'Average mRMSE',
                    'St Dev mRMSE', 'Average RMSE num', 'St Dev RMSE num', 'Average RMSE cat', 'St Dev RMSE cat', 
                    'Average PFC (%)', 'St Dev PFC (%)', 'Average execution time (s)', 'St Dev execution time (s)'])

for m in miss:
    for c in ctgan:
        for d in dat:
            if d == "news" and m == 0.3 and c !="":
                continue
            if m == 0.5 and c !="":
                continue

            average_m_rmse, st_dev_m_rmse, average_rmse_num, st_dev_rmse_num, average_rmse_cat, st_dev_rmse_cat, average_pfc, st_dev_pfc, average_exec_time, st_dev_exec_time = run_missforest(d, m, c, no_of_runs)
            results = {'Dataset': d, 'Missing%': m, 'Additional CTGAN data%': ctgan, 'Average mRMSE': average_m_rmse,
                    'St Dev mRMSE': st_dev_m_rmse, 'Average RMSE num': average_rmse_num, 'St Dev RMSE num': st_dev_rmse_num, 'Average RMSE cat': average_rmse_cat, 'St Dev RMSE cat': st_dev_rmse_cat, 
                    'Average PFC (%)': average_pfc, 'St Dev PFC (%)': st_dev_pfc, 'Average execution time (s)': average_exec_time, 'St Dev execution time (s)': st_dev_exec_time}
            df_results = df_results.append(results, ignore_index=True)

df_results.to_csv('imputed_data/summary_ctgan.csv', index=False)
            
            