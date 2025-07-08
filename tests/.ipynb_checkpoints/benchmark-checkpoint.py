import os
import time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, recall_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def run_iforest(bunch, params):
    x, y_ground_truth = bunch.data, bunch.target
    IF = IsolationForest(**params)
    
    # Fit + predict 
    start = time.perf_counter()
    IF.fit(x)
    y_pred = IF.predict(x) 
    elapsed = time.perf_counter() - start

    # Binairisation of the prediction and scores calculation 
    y_pred = (y_pred == -1)
    scores = -IF.decision_function(x)

    # metrics
    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    return {
        'auc': auc,
        'recall': recall,
        'time' : elapsed
    }


def run_lof(bunch, params) :
    x, y_ground_truth = bunch.data, bunch.target
    LOF = LocalOutlierFactor(** params)
    
    # Fit + predict 
    start = time.perf_counter()
    y_pred = LOF.fit_predict(x)
    elapsed = time.perf_counter() - start

    # Binairisation of the prediction and scores calculation 
    y_pred = (y_pred == -1)
    scores = -LOF.negative_outlier_factor_

    # metrics
    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    return {
        'auc': auc,
        'recall': recall,
        'time' : elapsed
    }
    


def benchmark(datasets, algorithms):
    auc_list, recall_list, time_list = [], [], []
    
    for bunch in datasets:
        auc_info, recall_info, time_info = {"dataset": bunch.name}, {"dataset": bunch.name}, {"dataset": bunch.name}
        
        for algo_name, algo_runner, params in algorithms:
            res = algo_runner(bunch, params) 
            
            auc_info[algo_name]    = res['auc']
            recall_info[algo_name] = res['recall']
            time_info[algo_name]   = res['time']
        print(bunch.name, " : fini")            
        auc_list.append(auc_info)
        recall_list.append(recall_info)
        time_list.append(time_info)


    df_auc    = pd.DataFrame(auc_list)
    df_recall = pd.DataFrame(recall_list)
    df_time   = pd.DataFrame(time_list)

    print(df_auc)
    
    return df_auc, df_recall, df_time

def pandas_to_latex(df_auc, df_recall, df_time):
    latex_auc = df_auc.to_latex(index=False,header=False,float_format="%.3f",line_terminator='\\\\ \\hline\n')
    latex_recall = df_recall.to_latex(index=False,header=False,float_format="%.3f",line_terminator='\\\\ \\hline\n')
    latex_time = df_time.to_latex(index=False,header=False,float_format="%.3f",line_terminator='\\\\ \\hline\n')

    print(latex_auc, latex_recall, latex_time)


def csv_data_into_bunch(filenames):
    """
    This function assume that the csv file will have label egal to 1 for anomalie and 0 for normal value.
    The csv have a header too.
    """
    datasets = []
    for i in filenames:
        name = os.path.splitext(os.path.basename(i))[0]

        data = pd.read_csv(i).to_numpy()
    
        x = data[:,:-1]
        y = data[:,-1]

        # Standardisation
        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(x)

        # Store in a Bunch
        datasets.append(Bunch(name=name, data=x_scaled, target=y))

    return datasets


        
def list_all_file_name_in_folder(folder="../data_rapport"):
    return [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".csv")]




if __name__ == "__main__":
    filenames = list_all_file_name_in_folder()
    print(filenames)
    
    datasets = csv_data_into_bunch(filenames)

    algorithms = [
     ("IForest", run_iforest, {'n_estimators':100, 'contamination':0.1}),
     ("LOF", run_lof, {'n_neighbors':150, 'contamination':0.1}),
    ]

    df_auc, df_recall, df_time = benchmark(datasets, algorithms)

    pandas_to_latex(df_auc, df_recall, df_time)