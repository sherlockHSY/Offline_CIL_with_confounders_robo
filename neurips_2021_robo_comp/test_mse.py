from re import T
import pandas as pd


from pathlib import Path
import glob
import pickle
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from copy import deepcopy
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
from oadam import OAdam, add_weight_decay, net_to_list
import joblib
import os

DictDataset = Dict[str, pd.DataFrame]

class LinearModel(nn.Module):
 
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim) 

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred    
    
    def predict(self, x):
        x = torch.from_numpy(x).float()
        y_pred = self.linear(x)
        
        return y_pred.detach().numpy()


class Model(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hidden_dim=128, softmax=False):
        super(Model, self).__init__()
        self.l1 = nn.Linear(in_dim, hidden_dim)
        self.l2 = nn.Linear(hidden_dim, 64)
        self.l3 = nn.Linear(64, out_dim)
        self.softmax = softmax

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        if self.softmax:
            return F.softmax(self.l3(x))
        return self.l3(x)


def get_state_columns(df: pd.DataFrame) -> List[str]:
    """Get names of columns related to the state of the system.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    List[str]
        List of state column names.
    """
    state_columns = [
        col for col in df.columns if col if col.startswith(("X", "Y", "d"))
    ]
    return state_columns


def get_input_columns(df: pd.DataFrame) -> List[str]:
    """Get names of columns related to the inputs given to the system.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input data.

    Returns
    -------
    List[str]
        List of input column names.
    """
    input_columns = [col for col in df.columns if col if col.startswith("U")]
    return input_columns


def sample_keys_test_val(
    dict_dataset: DictDataset, percentage_val=0.2
) -> Tuple[DictDataset, DictDataset, np.ndarray]:
    """Returns data split by test and validation trajectories.

    Parameters
    ----------
    dict_dataset : DictDataset
        Dataset with all trajectories.
    percentage_val : float, optional
        Percentage of trajectories to use in validation, by default 0.2

    Returns
    -------
    Tuple[DictDataset, DictDataset, np.ndarray]
        test and validation dataset and keys of trajectories defined for validation.
    """
    list_of_keys = list(dict_dataset.keys())
    validation_keys = np.random.choice(
        list_of_keys, size=int(len(dict_dataset) * percentage_val), replace=False
    )
    test_dict = {}
    validation_dict = {}

    for key in list_of_keys:
        if key in validation_keys:
            validation_dict[key] = dict_dataset[key]
        else:
            test_dict[key] = dict_dataset[key]
    return test_dict, validation_dict, validation_keys


# 包含历史坐标信息的数据集划分
def rs_convert_dict_to_np_dataset(
    dict_dataset: DictDataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts dictionary of datasets into numpy arrays used for testing the model.

    Parameters
    ----------
    dict_dataset : DictDataset
        Datasets organized by trajectory
    include_delta : bool, optional
        Whether to include the delta of positions between steps, by default False

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Trajectory data joined in a numpy array.
    """
    X = []
    X_past = []
    
    u = []
    for _, traj_df in dict_dataset.items():
        traj_df.fillna(0., inplace=True) 
        X.append(
                np.concatenate(
                    [
                        traj_df.iloc[1:-1][state_columns].to_numpy(),
                        traj_df.iloc[2:][next_positions].to_numpy(),
                        traj_df.iloc[2:][next_positions].to_numpy()
                            - traj_df.iloc[1:-1][next_positions].to_numpy(),
                    ],
                    axis=1,
                )
            )
        X_past.append(
                np.concatenate(
                    [
                        traj_df.iloc[:-2][state_columns].to_numpy(),
                        traj_df.iloc[1:-1][next_positions].to_numpy(),
                        traj_df.iloc[1:-1][next_positions].to_numpy()
                            - traj_df.iloc[:-2][next_positions].to_numpy(),
                    ],
                    axis=1,
                )
            )
        
        u.append(traj_df.iloc[1:-1][control_input_columns].to_numpy())
        
        
    u = np.concatenate(u, axis=0)
    X_arr = np.concatenate(X, axis=0)
    X_past_arr = np.concatenate(X_past, axis=0)
    return X_arr, X_past_arr, u


is_noisy = True

if is_noisy:
    num_traj = 50
    seed = 10
    algos = ['my_1','linear_noisy','rs_linear_noisy']
    # 读取测试集
    test_data_path = 'output/robo_test_noisy_random_0.5_10/'
    systems_path = f"output/robo_test_noisy_random_0.5_10/systems"
   
else:
    num_traj = 50
    seed = 10
    algos = ['linear', 'rs_linear']
    # 读取测试集
    test_data_path = 'output/robo_test_10/'
    systems_path = f"output/robo_test_10/systems"

models_path_n = f'models_{num_traj}_s_{seed}'
# 加载训练好的模型

with open(systems_path, "r") as f:
    systems_list = [i.strip() for i in f.readlines()]


for algo in algos:

    print('test algo: ', algo, ' num of traj: ', num_traj, ' seed: ', seed)
    models_path = f'data/ROBO/controllers/{algo}/controller/' 
    models_path = models_path + models_path_n

    with open(f"{models_path}/clip_y", "rb") as f:
        clip_y = pickle.load(f)

    with open(f"{models_path}/system_name_info", "rb") as f:
        system_info = pickle.load(f)


    total_loss = 0
    beetle_loss = 0
    bumblebee_loss = 0
    butterfly_loss = 0

    i_count  = 0
    for system in systems_list:
                
        data = {}
        if is_noisy:
            pattern = test_data_path + 'noisy_' + system + '_' + '*' + '.csv'
        else:
            pattern = test_data_path + 'noisy_' + system + '_' + '*' + '.csv'
        for file in glob.glob(
            pattern
        ):
            data[file[-6:-4]] = pd.read_csv(file)
        # Getting the first trajectory to explore columns
        traj1 = data["10"]
        state_columns = get_state_columns(traj1)
        control_input_columns = get_input_columns(traj1)
        next_positions = ["X", "Y"]
        INCLUDE_DELTA = True

        # Splitting the dicts into test-val sets.
        # Use all data (do not split) to test models used for submission.
        # Define the testing and test sets:
        test_dict, validation_dict, validation_keys = sample_keys_test_val(data, percentage_val=0.)
        
        n_inputs = len(state_columns)
        n_outputs = len(control_input_columns)

        i_count += 1
        
        # print('{} testing with {}, system: {}'.format(i_count, algo, system))

        # 测试集
        X_test, X_past, y_test = rs_convert_dict_to_np_dataset(test_dict)
        # print("state shape: ({},{}), action shape: ({},{}) ".format(X_test.shape[0],X_test.shape[1], y_test.shape[0],y_test.shape[1]))

        # Get the rotation matrix and its inverse for the input transformation:
        R, _, Rinv = np.linalg.svd(np.cov(y_test, rowvar=False))
        # y_test = np.matmul(y_test, R)

        # data process
        poly = PolynomialFeatures(2)

        scaler_path = f"{models_path}/{system}_scaler"
        with open(scaler_path, "rb") as f:
            scaler = pickle.load(f)

        model_input = scaler.transform(X_test)
        model_input = poly.fit_transform(model_input) 

        r_save_path = f"{models_path}/{system}_model_clipping_R"
        with open(r_save_path, "rb") as f:
            Rinv = pickle.load(f)

        n_inputs = system_info[system]['n_inputs']
        n_outputs = system_info[system]['n_outputs']

        model_save_path = f"{models_path}/{system}_model"
        model = LinearModel(n_inputs, n_outputs)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()

        loss_func = nn.MSELoss()
        y_test_pred = np.clip(model.predict(model_input),
                        clip_y[system]['min'],
                        clip_y[system]['max'])
        y_test_pred = np.matmul(y_test_pred, Rinv)
        mse_loss = loss_func(torch.from_numpy(y_test_pred), torch.from_numpy(y_test))
        mse_loss = mse_loss.detach().numpy()

        total_loss += mse_loss

        if system.endswith("beetle"):
            beetle_loss += mse_loss
        elif system.endswith("butterfly"):
            butterfly_loss += mse_loss
        elif system.endswith("bumblebee"):
            bumblebee_loss += mse_loss

    # 测试mse loss
    print(
            f"total mse loss for all system is :", round(total_loss, 3),
            '\n'
            f"beetle mse loss for system beetle is :", round(beetle_loss,3),
            '\n'
            f"butterfly mse loss for system butterfly is :", round(butterfly_loss,3),
            '\n'
            f"bumblebee mse loss for system bumblebee is :", round(bumblebee_loss,3),
            '\n'
        )