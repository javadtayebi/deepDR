import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def read_params(file_name):
    """
    Function for reading parameters of params file
    :param file_name:
    :return: params
    """

    params_df = pd.read_csv(file_name, sep='=', header=None)
    params_arr = np.array(params_df)

    params = {}
    for i in range(len(params_arr)):
        key = params_arr[i, 0]
        key = str(key.strip())

        val = params_arr[i, 1]
        val = str(val.strip())

        if key == 'select_nets':
            params[key] = list(map(int, val.strip('[]').split(',')))
        elif key == 'noise_factor':
            params[key] = float(val)
        elif key == 'ae_type':
            params[key] = str(val)
        else:
            params[key] = int(val)

    return params


def prepare_inputs(X, noise_factor=0.5, std=1.0):
    """
    Function for Corrupting the input to build noisy input and then train_test_split
    :param X:
    :param noise_factor:
    :param std:
    :return: X_train, X_train_noisy, X_test, X_test_noisy
    """

    # Corrupting the input to build noisy input
    if isinstance(X, list):
        Xs = train_test_split(*X, test_size=0.2)

        X_train = []
        X_test = []
        for i in range(0, len(Xs), 2):
            X_train.append(Xs[i])
            X_test.append(Xs[i + 1])
        X_train_noisy = list(X_train)
        X_test_noisy = list(X_test)

        for j in range(0, len(X_train)):
            X_train_noisy[j] = X_train_noisy[j] + noise_factor * np.random.normal(loc=0.0, scale=std,
                                                                                  size=X_train[j].shape)
            X_test_noisy[j] = X_test_noisy[j] + noise_factor * np.random.normal(loc=0.0, scale=std,
                                                                                size=X_test[j].shape)
            X_train_noisy[j] = np.clip(X_train_noisy[j], 0, 1)
            X_test_noisy[j] = np.clip(X_test_noisy[j], 0, 1)

    else:
        X_train, X_test = train_test_split(X, test_size=0.2)
        X_train_noisy = X_train.copy()
        X_test_noisy = X_test.copy()
        X_train_noisy = X_train_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=X_train.shape)
        X_test_noisy = X_test_noisy + noise_factor * np.random.normal(loc=0.0, scale=std, size=X_test.shape)
        X_train_noisy = np.clip(X_train_noisy, 0, 1)
        X_test_noisy = np.clip(X_test_noisy, 0, 1)

    return X_train, X_train_noisy, X_test, X_test_noisy


def compute_metric_mean(history, metric: str):
    history_keys = list(history.history.keys())

    history_metric_keys = []
    for i in range(len(history_keys)):
        if history_keys[i].endswith(metric):
            history_metric_keys.append(history_keys[i])

    metric = []
    val_metric = []
    for j in range(len(history_metric_keys)):
        if history_metric_keys[j].startswith('val'):
            val_metric.append(history.history[history_metric_keys[j]])
        else:
            metric.append(history.history[history_metric_keys[j]])

    return np.mean(metric, axis=0), np.mean(val_metric, axis=0)