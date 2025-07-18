import os
import time
import sys
import argparse
from datetime import datetime

import numpy as np
import pandas as pd

from python.algorithms import *
from python.creation import list_all_file_name_in_folder,csv_data_into_bunch, create_checkpoint_csv, append_dict_to_csv, create_checkpoint_csv2, load_variables_from_yaml


from sklearn.metrics import roc_curve, auc, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def benchmark_dataset_algo(datasets, algorithms, output_folder_checkpoints):
    """
    Run a benchmark of multiple algorithms on multiple datasets with checkpoints in csv.
    Apply all algorithms to each dataset in turn 
           
    
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

    create_checkpoint_csv(algorithms, output_folder_checkpoints) ### <- <-
    
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

        append_dict_to_csv(auc_info,os.path.join(output_folder_checkpoints,"auc_checkpoint.csv")) 
        append_dict_to_csv(recall_info,os.path.join(output_folder_checkpoints,"recall_checkpoint.csv"))
        append_dict_to_csv(time_info,os.path.join(output_folder_checkpoints,"time_checkpoint.csv"))
           
        print(bunch.name, " : fini")

    df_auc    = pd.DataFrame(auc_list)
    df_recall = pd.DataFrame(recall_list)
    df_time   = pd.DataFrame(time_list)
    
    return df_auc, df_recall, df_time

def benchmark_algo_dataset(datasets, algorithms, output_folder_checkpoints):
    """
    Run a benchmark of multiple algorithms on multiple datasets.
    Apply all datasets to each algorithm in turn.
    
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

    create_checkpoint_csv2(datasets, output_folder_checkpoints)
    auc_list, recall_list, time_list = [], [], []
    
    for algo_name, algo_runner, params in algorithms:
        print(algo_name, " : départ")
        auc_info, recall_info, time_info    = {"algorithm": algo_name}, {"algorithm": algo_name},{"algorithm": algo_name}        
        
        for bunch in datasets:
        
            print("  ", bunch.name, ": départ")
            res = algo_runner(bunch, params) 
            
            auc_info[bunch.name]    = res["auc"]
            recall_info[bunch.name] = res["recall"]
            time_info[bunch.name]   = res["time"]

            print("  ", "AUC :", res['auc'], "RAPPEL :", res['recall'],"TEMPS :", res['time'])
            print("  ", bunch.name, ": fini")
            
        auc_list.append(auc_info)
        recall_list.append(recall_info)
        time_list.append(time_info)
        print(algo_name, " : fini")

        append_dict_to_csv(auc_info,os.path.join(output_folder_checkpoints,"auc_checkpoint.csv")) 
        append_dict_to_csv(recall_info,os.path.join(output_folder_checkpoints,"recall_checkpoint.csv"))
        append_dict_to_csv(time_info,os.path.join(output_folder_checkpoints,"time_checkpoint.csv"))
        
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
    
def pandas_to_latex_with_ranks(df, ascending):
    """
    Convert a df to latex, adding per-row rankings and bolding the top value.

    Parameters
    ----------
    df : pandas 
    ascending : bool
        If True, rank 1 = smallest value; if False, rank 1 = largest value.

    Returns
    -------
    str
    """
    # Identify numeric columns to rank, and preserve all others
    num_cols = df.select_dtypes(include="number").columns
    other_cols = [c for c in df.columns if c not in num_cols]

    # Round values to two decimals 
    vals = df[num_cols].round(2)

    # Per row ranks, direction controlled by ascending
    ranks = vals.rank(axis=1, method='dense', ascending=ascending).astype(int)

    # Cconvert to strings, because the result will go in this dataset 
    vals = vals.astype(str)

    # Combine each value with its rank, bolding n.1 
    for idx in df.index:
        for col in num_cols:
            v = vals.at[idx, col]
            r = ranks.at[idx, col]
            
            if r == 1: 
                cell = f"\\textbf{{{v}}} ({r})"
            else:
                cell = f"{v} ({r})"
            vals.at[idx, col] = cell

    df_final = pd.concat([df[other_cols], vals], axis=1)
    
    return add_hlines(df_final.to_latex(index=False, float_format="%.2f"))


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



def automatisation(mode, datasets, algorithms, filepath = "./latex_results.txt", output_folder_checkpoints="./checkpoints/"):   
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
    print("DÉBUT, ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print(algorithms)
    start = time.perf_counter()

    # Calculate all the metrics
    if mode == "da":
        df_auc, df_recall, df_time = benchmark_dataset_algo(datasets, algorithms, output_folder_checkpoints)
    else:
       df_auc, df_recall, df_time = benchmark_algo_dataset(datasets, algorithms, output_folder_checkpoints) 

    # Transform the metrics into latex tab 
    latex_auc = pandas_to_latex_with_ranks(df_auc, False)
    latex_recall = pandas_to_latex_with_ranks(df_recall, False)
    latex_time = pandas_to_latex_with_ranks(df_time, True)

    # Write these latex tab in a file
    write_latex_in_file(latex_auc, latex_recall,latex_time, algorithms, filepath)
    elapsed = time.perf_counter() - start
    
    print("FIN apres ", elapsed, "secondes !!!")


def load_datasets(dataset_folder_path,names=None):
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
    
    all_files = list_all_file_name_in_folder(dataset_folder_path)
    if names:
        # Keep only the datasets whose  appears in the provided names list.
        selected_files = [f for f in all_files 
                          if os.path.splitext(os.path.basename(f))[0].lower() in names ]
    else:
        # Or select all datasets
        selected_files = all_files

    print("Datasets used :", selected_files)
    # Convert the selected csv files into a list of bunch objects.
    return csv_data_into_bunch(selected_files)

def get_algorithms(all_algos, selected=None):
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
    
    if selected:
        # Filter the full list to only include those is in the provided selected list 
        sel = {name.lower() for name in selected}
        selected_algo = [t for t in all_algos 
                if t[0].lower() in sel]
        print("Algortihms used :", sel)
        return selected_algo

    print("All algorithms used")
    return all_algos


def parse_args(algo_names,dataset_folder_path):
    """
    Parse command-line arguments to select datasets and algorithms.

    Returns
    -------
        Parser args:
        - dataset: list of chosen dataset names (or None)
        - algo:    list of chosen algorithm names (or None)
    """
    
    ds_names = [os.path.splitext(os.path.basename(p))[0] for p in list_all_file_name_in_folder(dataset_folder_path)]
    
    parser = argparse.ArgumentParser(description="Benchmark script for multiple anomaly detection algorithms")

    choices = [n.lower() for n in ds_names]

    # Choose execution order: 
    parser.add_argument(
        "-m",
        choices=["da","ad"],
        default="da",
        help= "Execution order:\n"
            "  'da (dataset-algo)'  — apply all algorithms to each dataset in turn (default)\n"
            "  'ad (algo-dataset)'  — apply all datasets to each algorithm in turn."
    )

    # Choose dataset(s): 
    parser.add_argument(
        "-d",
        nargs="+", # 1 or multiple datasets
        metavar=(choices,"Other dataset if you want"),
        choices=choices, # choices possible 
        type=str.lower, # convert the user in lowercase 
        help="Dataset(s) to load (use base filename, without .csv). Default: all."
    )

    # Choose algo(s): 
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
    dataset_folder_path, output_folder_checkpoints, latex_output, all_algos = load_variables_from_yaml()
    

    algo_names = [name.lower() for name, _, _ in all_algos]
    
    args = parse_args(algo_names, dataset_folder_path)
    datasets = load_datasets(dataset_folder_path, args.d)
    algorithms = get_algorithms(all_algos, args.a)
    mode = args.m
    print("mode: ", mode)
    

    automatisation(mode, datasets, algorithms, latex_output, output_folder_checkpoints)