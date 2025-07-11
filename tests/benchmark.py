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

from pyod.models.auto_encoder import AutoEncoder

        
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
    """
    Fit and evaluate IsolationForest on a single dataset.

    Parameters
    ----------
    bunch : Bunch
        Must have attributes:
          - data: feature data
          - target: ground truth labels
    params : dict
        Keyword arguments for IsolationForest constructor.

    Returns
    -------
    dict
        {
          'auc': float,      # ROC AUC score
          'recall': float,   # recall score
          'time': float      # elapsed time in seconds
        }
    """
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
    """
    Fit and evaluate LocalOutlierFactor on a single dataset.

    Parameters
    ----------
    bunch : Bunch
        Must have attributes:
          - data: feature data
          - target: ground truth labels
    params : dict
        Keyword arguments for LocalOutlierFactor constructor.

    Returns
    -------
    dict
        {
          'auc': float,      # ROC AUC score
          'recall': float,   # recall score
          'time': float      # elapsed time in seconds
        }
    """
    
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
    
def run_autoencoder(bunch, params):
    """
    Fit and evaluate an AutoEncoder pyOD on a single dataset.

    Parameters
    ----------
    bunch : Bunch
        Must have attributes:
          - data: feature data
          - target: ground truth labels
    params : dict
        Keyword arguments for IsolationForest constructor.

    Returns
    -------
    dict
        {
          'auc': float,      # ROC AUC score
          'recall': float,   # recall score
          'time': float      # elapsed time in seconds
        }
    """
    x, y_ground_truth = bunch.data, bunch.target
    AE = AutoEncoder(**params)

    # Fit + predict 
    start = time.perf_counter()
    AE.fit(x)
    y_pred = AE.predict(x) 
    elapsed = time.perf_counter() - start

    scores = AE.decision_function(x)

    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }

def benchmark(datasets, algorithms):
    """
    Run a benchmark of multiple algorithms on multiple datasets.
    
     datasets : list of Bunch
            Each Bunch must have attributes:
              - name : dataset name
              - data : feature data
              - target : ground truth labels
        algorithms : list of tuples
            Each tuple is (algo_name, runner_function, params_dict):
              - algo_name : descriptive name of the algorithm
              - runner_function : function(bunch, params) - > dict
              - params_dict: parameters to pass to the runner


    Returns
    -------
    df_auc : pandas.DataFrame
        AUC scores for each (dataset × algorithm).
    df_recall : pandas.DataFrame
        Recall scores for each (dataset × algorithm).
    df_time : pandas.DataFrame
        Execution time (seconds) for each (dataset × algorithm).
    """
    
    auc_list, recall_list, time_list = [], [], []
    
    for bunch in datasets:
        print(bunch.name, " : départ")
        auc_info, recall_info, time_info = {"dataset": bunch.name}, {"dataset": bunch.name}, {"dataset": bunch.name}
        
        for algo_name, algo_runner, params in algorithms:
            print("  ", algo_name, ": départ")
            res = algo_runner(bunch, params) 
            
            auc_info[algo_name]    = res['auc']
            recall_info[algo_name] = res['recall']
            time_info[algo_name]   = res['time']
            print("AUC :", res['auc'], "RAPPEL :", res['recall'],"TEMPS :", res['time'])
            print("  ", algo_name, ": fini")
            
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
    """
    Convert three pandas DataFrames into LaTeX-formatted tables with horizontal lines.

    Parameters
    ----------
    df_auc : pandas.DataFrame
        DataFrame of AUC scores.
    df_recall : pandas.DataFrame
        DataFrame of recall scores.
    df_time : pandas.DataFrame
        DataFrame of execution times.

    Returns
    -------
    tuple of str
        A tuple (latex_auc, latex_recall, latex_time), where each element is the
        LaTeX code for the corresponding DataFrame, with added \\hline commands.
    """
    latex_auc = add_hlines(df_auc.to_latex(index=False,float_format="%.2f"))
    latex_recall = add_hlines(df_recall.to_latex(index=False,float_format="%.2f"))
    latex_time = add_hlines(df_time.to_latex(index=False,float_format="%.2f"))

    return latex_auc, latex_recall, latex_time


def write_latex_in_file(latex_auc, latex_recall,latex_time, algorithms, filepath = "./latex_results.txt"):
    """
    Prepend LaTeX tables and algorithm info to a text file with a timestamp.

    Parameters
    ----------
    latex_auc : str
        LaTeX code for the AUC table.
    latex_recall : str
        LaTeX code for the recall table.
    latex_time : str
        LaTeX code for the time table.
    algorithms : list of tuples
        Same list passed to benchmark, each tuple is
        (name, runner_function, params_dict).
    filepath : str, optional
        Path to the output text file (default "./latex_results.txt").

    Notes
    -----
    - The current timestamp is written at the top.
    - Algorithm names and their parameter settings are listed.
    - New content is prepended to preserve previous runs.
    """
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
    """
    Run the full pipeline: benchmark algorithms, convert results to LaTeX, and write to file.

    Parameters
    ----------
    datasets : list of Bunch
        List of dataset objects (each with .name, .data, .target).
    algorithms : list of tuples
        Each tuple is (name, runner_function, params_dict), as for benchmark().
    filepath : str, optional
        Path where the LaTeX output will be written (default "./latex_results.txt").

    """
    print("DÉBUT")
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
    
    """
    algorithms = [
     ("IForest", run_iforest, {'n_estimators':200, 'contamination':0.1}),
     ("LOF", run_lof, {'n_neighbors':300, 'contamination':0.1}),
     ("AutoEncoder", run_autoencoder, {'contamination':0.1})
    ]
    """
    """
    algorithms = [
     ("IForest", run_iforest, {'n_estimators':100, 'contamination':0.1}),
     ("AutoEncoder", run_autoencoder, {'contamination':0.1})
    ]
    """
    
    algorithms = [
     ("IForest", run_iforest, {'n_estimators':100, 'contamination':0.1}),
     ("LOF", run_lof, {'n_neighbors':150, 'contamination':0.1}),
     ("AutoEncoder", run_autoencoder, {'contamination':0.1})
    ]
    
    

    automatisation(datasets, algorithms)