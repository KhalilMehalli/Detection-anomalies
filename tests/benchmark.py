import os
import time
from datetime import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc, recall_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

        
def list_all_file_name_in_folder(folder="../data_rapport"):
    return [os.path.join(folder, filename) for filename in os.listdir(folder) if filename.endswith(".csv")]

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
        print(bunch.name, " : départ")
        auc_info, recall_info, time_info = {"dataset": bunch.name}, {"dataset": bunch.name}, {"dataset": bunch.name}
        
        for algo_name, algo_runner, params in algorithms:
            res = algo_runner(bunch, params) 
            
            auc_info[algo_name]    = res['auc']
            recall_info[algo_name] = res['recall']
            time_info[algo_name]   = res['time']
            
        auc_list.append(auc_info)
        recall_list.append(recall_info)
        time_list.append(time_info)
        print(bunch.name, " : fini")
        
    df_auc    = pd.DataFrame(auc_list)
    df_recall = pd.DataFrame(recall_list)
    df_time   = pd.DataFrame(time_list)
    
    return df_auc, df_recall, df_time
        

def add_hlines(latex):
    return latex.replace("\\\\", "\\\\ \\hline")
        
def pandas_to_latex(df_auc, df_recall, df_time):

    latex_auc = add_hlines(df_auc.to_latex(index=False,float_format="%.2f"))
    latex_recall = add_hlines(df_recall.to_latex(index=False,float_format="%.2f"))
    latex_time = add_hlines(df_time.to_latex(index=False,float_format="%.2f"))

    return latex_auc, latex_recall, latex_time


def write_latex_in_file(latex_auc, latex_recall,latex_time, algorithms, filepath = "./latex_results.txt"):
    # Actual time
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Informations about algorithms used : name1 : param1 = 0, param2=0 ; ...
    infos = []
    for name, _, params in algorithms:
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        infos.append(f"{name}: {param_str}")
    info_algo = "; ".join(infos)

    content = (
        f"{now}\n"
        f"{info_algo}\n\n"
        r"%=== AUC table ===" "\n"
        f"{latex_auc}\n\n"
        r"%=== Recall table ===" "\n"
        f"{latex_recall}\n\n"
        r"%=== Time table ===" "\n"
        f"{latex_time}\n"
    )

    # Add at the start of the file without delete the old content 
    old = ""
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            old = f.read()
            
    with open(filepath, "w") as f:
            f.write(content)
            f.write(old)



def automatisation(datasets, algorithms, filepath = "./latex_results.txt"):   
    # Calculate all the metrics
    df_auc, df_recall, df_time = benchmark(datasets, algorithms)

    # Transform the metrics into latex tab 
    latex_auc, latex_recall, latex_time = pandas_to_latex(df_auc, df_recall, df_time)

    # Write these latex tab in a file
    write_latex_in_file(latex_auc, latex_recall,latex_time, algorithms, filepath)
    print("FIN")


if __name__ == "__main__":
    filenames = list_all_file_name_in_folder()
    
    datasets = csv_data_into_bunch(filenames)
 
    algorithms = [
     ("IForest", run_iforest, {'n_estimators':200, 'contamination':0.1}),
     ("LOF", run_lof, {'n_neighbors':300, 'contamination':0.1}),
    ]


    automatisation(datasets, algorithms)