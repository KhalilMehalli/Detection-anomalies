import os
import time
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.metrics import roc_curve, auc, recall_score, roc_auc_score
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch

from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ae1svm import AE1SVM
from pyod.models.knn import KNN
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

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)

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

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
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
    
    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }

def run_vae(bunch, params):
    x, y_ground_truth = bunch.data, bunch.target
    VAE_model = VAE(**params)

    # Fit + predict 
    start = time.perf_counter()
    VAE_model.fit(x)
    y_pred = VAE_model.predict(x)
    elapsed = time.perf_counter() - start

    # metrics
    scores = VAE_model.decision_function(x)
    
    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }



def run_deepsvdd(bunch, params):
    x, y_ground_truth = bunch.data, bunch.target

    # DeepSVDD have a mandatory parameters, the features numbers of the dataset      <- <- <-
    features = x.shape[1]
    params["n_features"] = features
    
    DS = DeepSVDD(**params)

    # Fit + predict 
    start = time.perf_counter()
    DS.fit(x)
    y_pred = DS.predict(x)
    elapsed = time.perf_counter() - start

    # metrics 
    scores = DS.decision_function(x)
    
    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }

def run_ae1svm(bunch, params):
    x, y_ground_truth = bunch.data, bunch.target

    AE1SVM_model = AE1SVM(**params)

    # Fit + predict 
    start = time.perf_counter()
    AE1SVM_model.fit(x)
    y_pred = AE1SVM_model.predict(x)
    elapsed = time.perf_counter() - start

    # metrics
    scores = AE1SVM_model.decision_function(x)
    
    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }

def run_knn(bunch, params):
    x, y_ground_truth = bunch.data, bunch.target
    KNN_model = KNN(**params)

    # Fit + predict 
    start = time.perf_counter()
    KNN_model.fit(x)
    y_pred = KNN_model.predict(x)
    elapsed = time.perf_counter() - start

    # metrics
    scores = KNN_model.decision_function(x)
    
    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
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
    start = time.perf_counter()

    # Calculate all the metrics
    df_auc, df_recall, df_time = benchmark(datasets, algorithms)

    # Transform the metrics into latex tab 
    latex_auc, latex_recall, latex_time = pandas_to_latex(df_auc, df_recall, df_time)

    # Write these latex tab in a file
    write_latex_in_file(latex_auc, latex_recall,latex_time, algorithms, filepath)
    elapsed = time.perf_counter() - start
    
    print("FIN apres ", elapsed, "secondes !!!")


def load_datasets(names=None):
    """
    Load only the requested datasets into memory as lists of unch objects.

    Parameters
    ----------
    names : list of str or None
        If provided, contains the base filenames (without .csv) to load.
        If None, all CSV files are loaded.

    Returns
    -------
    list of bunch
        Each bunch has attributes .name, .data, .target.
    """
    
    all_files = list_all_file_name_in_folder()
    if names:
        # Keep only the datasets whose  appears in the provided names list.
        selected_files = [f for f in all_files 
                          if os.path.splitext(os.path.basename(f))[0] in names ]
    else:
        # Or select all datasets
        selected_files = all_files

    print("Datasets used :", selected_files)
    # Convert the selected csv files into a list of bunch objects.
    return csv_data_into_bunch(selected_files)

def get_algorithms(selected=None):
    """
    Selected only the algorithms the user want.

    Parameters
    ----------
    selected : list of str or None
        If provided, contains the algorithm names 
        to include. If None, all supported algorithms are returned.
        
    Returns
    -------
    list of tuples
        Each tuple is (str, function, dict).
    """
    
    all_algos = [
        ("IForest",     run_iforest,     {"random_state":10}),
        ("LOF",         run_lof,         {}),
        ("AE", run_autoencoder, {"random_state":10, "preprocessing": False,"epoch_num": 10}),
        ("VAE",         run_vae,         {"random_state":10, "preprocessing": False,"epoch_num": 10}),
        ("DeepSVDD",    run_deepsvdd,    {"random_state":10,"preprocessing": False,"epochs": 10}),
        ("AE1SVM",      run_ae1svm,      {"preprocessing": False,"epochs": 10}),
        ("KNN",         run_knn,         {})
    ]

    if selected:
        # Filter the full list to only include those is in the provided selected list 
        sel = {name.lower() for name in selected}
        selected_algo = [t for t in all_algos 
                if t[0].lower() in sel]
        print("Algortihms used :", sel)
        return selected_algo

    print("All algorithms used")
    return all_algos


def parse_args():
    """
    Parse command-line arguments to select datasets and algorithms.

    Returns
    -------
    Namespace
        Attributes:
        - dataset: list of chosen dataset names (or None)
        - algo:    list of chosen algorithm names (or None)
    """
    
    ds_names = [os.path.splitext(os.path.basename(p))[0] for p in list_all_file_name_in_folder()]
    algo_names = ['iforest','lof','ae','vae','deepsvdd','ae1svm','knn']
    
    parser = argparse.ArgumentParser(description="Benchmark script for multiple anomaly detection algorithms")

    choices = [n.lower() for n in ds_names]
                                
    parser.add_argument(
        "-d",
        nargs="+", # 1 or multiple datasets
        metavar=(choices,"Other dataset if you want"),
        choices=choices, # choices possible 
        type=str.lower, # convert the user in lowercase 
        help="Dataset(s) to load (use base filename, without .csv). Default: all."
    )
    parser.add_argument(
        "-a",
        nargs="+",
        metavar=(algo_names,"Other algo if you want"),
        choices=algo_names,
        type=str.lower,
        help="Algorithm(s) to run. Default: all."
    )
    return parser.parse_args()
    
if __name__ == "__main__":
    
    args = parse_args()
    datasets = load_datasets(args.d)
    algorithms = get_algorithms(args.a)


    automatisation(datasets, algorithms)