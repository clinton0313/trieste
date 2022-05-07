import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from trieste.objectives.utils import mk_observer
from trieste.data import Dataset
from trieste.types import TensorType
from typing import Callable

def build_objective(X: pd.DataFrame, Y: pd.DataFrame, model = SVR, **kwargs) -> Callable:
    def objective_function(query_point: TensorType) -> float:
        hyperparameters = query_point.numpy()
        svm = model(gamma = hyperparameters[0], c = hyperparameters[1], epsilon = hyperparameters[2])
        score = cross_val_score(svm, X, Y, **kwargs)
        return score
    
    return mk_observer(objective_function)

def exhaustive_search(obvserver: Callable, hyperparameters_range: TensorType, grid_size: int) -> TensorType:
    """
    :param hyperparameters_range: dx2 for d hyperparameters (min, max)
    :param observer: observer function of the cross_val_score
    :param grid_size: number of values to try for each hyperparameter
    return dx1 hyperparameters
    """

    param_grid = [np.linspace(hyper[0], hyper[1], grid_size) for hyper in hyperparameters_range]