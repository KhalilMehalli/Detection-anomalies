import time
from sklearn.metrics import roc_curve, auc, recall_score, roc_auc_score, precision_score, confusion_matrix, classification_report

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from pyod.models.vae import VAE
from pyod.models.deep_svdd import DeepSVDD
from pyod.models.ae1svm import AE1SVM
from pyod.models.knn import KNN
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.anogan import AnoGAN
from pyod.models.ocsvm import OCSVM



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

    print("fit + predict fini")	
    # Binairisation of the prediction and scores calculation 
    y_pred = (y_pred == -1)
    scores = -IF.decision_function(x)

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))


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

    print("fit + predict fini")	

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))

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

    print("fit + predict fini")	

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))

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

    print("fit + predict fini")	
    # metrics

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))

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

    print("fit + predict fini")	
    # metrics 

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))

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

    print("fit + predict fini")	
    # metrics

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))

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

    print("fit + predict fini")	
    # metrics
    scores = KNN_model.decision_function(x)
    
    auc = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)
    
    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))
    
    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }

def run_anogan(bunch, params):
    x, y_ground_truth = bunch.data, bunch.target
    AG = AnoGAN(**params)

    # Fit + predict
    start = time.perf_counter()
    AG.fit(x)
    y_pred = AG.predict(x)
    elapsed = time.perf_counter() - start
    
    print("fit + predict fini")	
    # metrics
    scores = AG.decision_function(x)

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))

    auc    = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }

def run_ocsvm(bunch, params):
    x, y_ground_truth = bunch.data, bunch.target
    OC = OCSVM(**params)

    # Fit + predict
    start = time.perf_counter()
    OC.fit(x)
    y_pred = OC.predict(x)
    elapsed = time.perf_counter() - start
    
    print("fit + predict fini")	
    # metrics

    print(confusion_matrix(y_ground_truth, y_pred))
    print(classification_report(y_ground_truth, y_pred), sep="\n")
    print("précision", precision_score(y_ground_truth, y_pred))

    scores = OC.decision_function(x)

    auc    = roc_auc_score(y_ground_truth, scores)
    recall = recall_score(y_ground_truth, y_pred)

    total_time = time.perf_counter() - start
    print("Total execution time", total_time)
    
    return {
        'auc':    auc,
        'recall': recall,
        'time':   elapsed
    }

