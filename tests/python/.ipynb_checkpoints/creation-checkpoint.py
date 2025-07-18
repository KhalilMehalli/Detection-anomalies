import os
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch
from sklearn.metrics import roc_curve, auc

import yaml
import importlib

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


def barplot(scores,y_ground_truth, color_outlier='red', color_inlier='green', title="Outliers scores " ):
    mask_outlier = (y_ground_truth==1)
    mask_inlier = ~mask_outlier

    # plt.bar need an indices tab
    indices = np.arange(scores.size)

    start = time.perf_counter()
    plt.bar(indices[mask_outlier], scores[mask_outlier], color=color_outlier, edgecolor=color_outlier)
    plt.bar(indices[mask_inlier], scores[mask_inlier], color=color_inlier, edgecolor=color_inlier)
    elapsed = time.perf_counter() - start
    print(f"Plot execution time : {elapsed:.4f} s")
    

    plt.xlabel("index points")
    plt.ylabel("Outliers scores")
    plt.title(title)


def plot_point(x,y,title):
    mask_outlier = (y==1)
    mask_inlier = ~mask_outlier
    plt.scatter(x[mask_inlier, 0], x[mask_inlier, 1], color="green", label="Inliers")
    plt.scatter(x[mask_outlier, 0], x[mask_outlier, 1], color="red", label="Outliers")
    plt.legend(bbox_to_anchor=(1.02, 1)) # Legend out of the plot if it will hide points
    plt.title(title)

def plot_roc(y_ground_truth, scores,title="ROC Curve"):
    fpr, tpr, _ = roc_curve(y_ground_truth, scores)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f"{title} (AUC = {roc_auc:.3f})")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend(loc="lower right")
    plt.title(title)


def harmonize_labels(filenames,outlier_label='o',header=None):
    for path in filenames:
        df = pd.read_csv(path, header=header)
        
        label_col = df.columns[-1]
        
        df[label_col] = (df[label_col] == outlier_label).astype(int)
        
        df.to_csv(path, index=False)

def csv_data_infos(filenames, outlier_label=1):
    for name in filenames:
        data = pd.read_csv(name).to_numpy()
        
        x = data[:,:-1]
        y = data[:,-1]
        
        samples = x.shape[0]
        features = x.shape[1]

        # Transform label for outlier into 1 and 0 for the normal value
        y_bin = (y == outlier_label).astype(int)

        anomaly = np.sum(y_bin)
        p_anomaly = anomaly/ samples

        print({
            'filename' : name,
            'samples'  : samples,
            'features' : features,
            'anomalies': anomaly,
            'p_anomaly': p_anomaly
        })


def create_checkpoint_csv(algorithms, output_folder_checkpoints):
    os.makedirs(output_folder_checkpoints, exist_ok=True) # Create the repertories 
    algo_names  = [name for name, _, _ in algorithms]
    cols = ["dataset"] + algo_names
    
    # Create empty CSVs with header row
    pd.DataFrame(columns=cols).to_csv(   os.path.join(output_folder_checkpoints,"auc_checkpoint.csv"),    mode="w", index=False)
    pd.DataFrame(columns=cols).to_csv(  os.path.join(output_folder_checkpoints,"auc_checkpoint.csv") , mode="w", index=False)
    pd.DataFrame(columns=cols).to_csv( os.path.join(output_folder_checkpoints,"auc_checkpoint.csv")  ,   mode="w", index=False)

def append_dict_to_csv(row, path): 
    """
        Add a row in a csv file, I use pandas because it's easy to write and I have a small number of dataset
    """
    pd.DataFrame([row]).to_csv(path, mode="a", header=False, index=False, float_format="%.2f")

def create_checkpoint_csv2(datasets, output_folder_checkpoints ):
    os.makedirs(output_folder_checkpoints, exist_ok=True) # Create the repertories 
    dataset_names = [b.name for b in datasets]
    cols = ["algorithm"] + dataset_names

    # write empty CSVs with that header (mode="w" to overwrite)
    pd.DataFrame(columns=cols).to_csv(   os.path.join(output_folder_checkpoints,"auc_checkpoint.csv"),    mode="w", index=False)
    pd.DataFrame(columns=cols).to_csv(  os.path.join(output_folder_checkpoints,"auc_checkpoint.csv") , mode="w", index=False)
    pd.DataFrame(columns=cols).to_csv( os.path.join(output_folder_checkpoints,"auc_checkpoint.csv")  ,   mode="w", index=False)
    
def load_variables_from_yaml(path="./infos.yaml"):
    """
    Load a yaml file and return usable variables:
    Parameters
    ----------
    path : path of the yaml file
    algo_file = path of the file which contains all the algo

    Returns
    -------
    - datasets_folder : path to the datasets folder
    - output_folder_checkpoints : path to the folder which store CSV checkpoints
    - latex_output : path to the latex results file
    - algorithms (list of tuples): (name, runner_function, params_dict)
    """
    # Load YAML file 
    with open(path, "r") as f:
        info = yaml.safe_load(f)

    # Extract info 
    datasets_folder = info["datasets_folder"]
    output_folder_checkpoints = info.get("output_folder_checkpoints", "./checkpoints/")
    latex_output = info.get("latex_output", "./latex_results.txt")
    algo_file = info["algo_file"]

    # Dynamically import the file where runner functions are defined
    algo_module = importlib.import_module(algo_file)  # import the file python.algorithms.py

    # Build the list of algorithm tuples
    algorithms = []
    for algo in info.get("algorithms", []):
        name = algo["name"]
        runner_fn = getattr(algo_module, algo["runner"])
        params = algo.get("params", {})
        algorithms.append((name, runner_fn, params))

    return datasets_folder, output_folder_checkpoints, latex_output, algorithms
    