import argparse
import sys
from lbd_comp.evaluate_track2 import evaluate_track2

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


def sample_keys_train_val(
    dict_dataset: DictDataset, percentage_val=0.2
) -> Tuple[DictDataset, DictDataset, np.ndarray]:
    """Returns data split by train and validation trajectories.

    Parameters
    ----------
    dict_dataset : DictDataset
        Dataset with all trajectories.
    percentage_val : float, optional
        Percentage of trajectories to use in validation, by default 0.2

    Returns
    -------
    Tuple[DictDataset, DictDataset, np.ndarray]
        Train and validation dataset and keys of trajectories defined for validation.
    """
    list_of_keys = list(dict_dataset.keys())
    validation_keys = np.random.choice(
        list_of_keys, size=int(len(dict_dataset) * percentage_val), replace=False
    )
    train_dict = {}
    validation_dict = {}

    for key in list_of_keys:
        if key in validation_keys:
            validation_dict[key] = dict_dataset[key]
        else:
            train_dict[key] = dict_dataset[key]
    return train_dict, validation_dict, validation_keys


# 包含历史坐标信息的数据集划分
def rs_convert_dict_to_np_dataset(
    dict_dataset: DictDataset
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts dictionary of datasets into numpy arrays used for training the model.

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

# 原先的数据集划分
def convert_dict_to_np_dataset(
    dict_dataset: DictDataset, include_delta=True
) -> Tuple[np.ndarray, np.ndarray]:
    """Converts dictionary of datasets into numpy arrays used for training the model.

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
    y = []
    u = []
    for _, traj_df in dict_dataset.items():
        if include_delta:
            X.append(
                np.concatenate(
                    [
                        traj_df.iloc[:-1][state_columns].to_numpy(),
                        traj_df.iloc[1:][next_positions].to_numpy(),
                        traj_df.iloc[1:][next_positions].to_numpy()
                        - traj_df.iloc[:-1][next_positions].to_numpy(),
                        # traj_df.iloc[:-1][control_input_columns].to_numpy(),
                    ],
                    axis=1,
                )
            )
        else:
            X.append(
                np.concatenate(
                    [
                        traj_df.iloc[:-1][state_columns].to_numpy(),
                        traj_df.iloc[1:][next_positions].to_numpy(),
                        # traj_df.iloc[:-1][control_input_columns].to_numpy(),
                    ],
                    axis=1,
                )
            )
        u.append(traj_df.iloc[:-1][control_input_columns].to_numpy())
        y.append(traj_df.iloc[1:][state_columns].to_numpy())
    u = np.concatenate(u, axis=0)
    X_arr = np.concatenate(X, axis=0)
    y_arr = np.concatenate(y, axis=0)
    return X_arr, u

# ==========================================================================
def init_weights(m):
    if type(m) == nn.Linear:
        m.weight.data.uniform_(-1.0, 1.0)
        m.bias.data.uniform_(-1.0, 1.0)

# 线性模型 torch版
class LinearModel(nn.Module):
 
    def __init__(self, in_dim, out_dim):
        super(LinearModel, self).__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim) 

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred    
    
    def predict(self, x):
        x = torch.from_numpy(x).float().to(device)
        y_pred = self.linear(x)
        
        return y_pred.cpu().detach().numpy()

# 三层神经网络
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
    
    def predict(self, x):
        x = torch.from_numpy(x).float().to(device)
        y_pred = self.forward(x)
        
        return y_pred.cpu().detach().numpy()

# 梯度下降版线性回归
def LR_GD(n_inputs, n_outputs, X, y, lr=1e-3, batch_size=128, epochs=1e4):
    nn_model = LinearModel(n_inputs, n_outputs).to(device)
    loss_func = nn.MSELoss()
    optimizer = optim.Adam(nn_model.parameters(), lr=lr)
    # optimizer = OAdam(net_to_list(nn_model),
    #                                     lr=lr, betas=(0, .01))
    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    for e in range(int(epochs)):
        # train
        idx = np.random.choice(len(X), batch_size)
        
        X_batch = X
        y_batch = y
        pred_y = nn_model.forward(X_batch)

        # Compute and print loss
        loss = loss_func(pred_y, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return nn_model


if __name__ == "__main__":
    # get param
    parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=10,
        help="choose seed",
    )
    parser.add_argument(
        "-n",
        "--traj_num",
        type=int,
        default=50,
        help="traj_num",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        type=int,
        default=0,
        help="whether to use gpu, 0 mean use cpu, 1 mean use gpu",
    )

    input_args = sys.argv[1:]
    args = parser.parse_args(input_args)

    seed = args.seed
    num_traj = args.traj_num

    seed_arr = [100]
    for seed_i in seed_arr:
        seed = seed_i
        print('seed: ', seed)
        # 设置随机种子
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

        # set device to cpu or cuda
        cuda = args.cuda
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:'+str(cuda))
            # torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")

        # num_arr = [10,20,30,40]
        num_arr = [30,40,50,60,70]
        for i_num in num_arr:
            num_traj = i_num
            print("当前训练的轨迹数：", num_traj)
            algo = 'linear_noisy'
            model_path = f"data/ROBO/controllers/{algo}/controller/models_{num_traj}_s_{seed}/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)


            # noisy datas
            systems_path = f"output/robo_train_noisy_random_0.5_{num_traj}/systems"
            data_path = f'output/robo_train_noisy_random_0.5_{num_traj}/'

            # systems_path = "data/ROBO/training_trajectories/systems"
            # data_path = 'data/ROBO/training_trajectories/'

            with open(systems_path, "r") as f:
                systems_list = [i.strip() for i in f.readlines()]

            system_name_info = {}
            clip_y = {}
            i_count = 0
            scalers = {}
            models = defaultdict(dict)
            for system_name in systems_list:
                
                data = {}
                pattern = data_path + system_name + '_' + '*' + '.csv'
                
                for file in glob.glob(
                    pattern
                ):
                    data[file[-6:-4]] = pd.read_csv(file)
                # Getting the first trajectory to explore columns
                traj1 = data["00"]
                state_columns = get_state_columns(traj1)
                control_input_columns = get_input_columns(traj1)
                next_positions = ["X", "Y"]
                INCLUDE_DELTA = True

                # Splitting the dicts into train-val sets.
                # Use all data (do not split) to train models used for submission.
                # Define the training and test sets:
                train_dict, validation_dict, validation_keys = sample_keys_train_val(data, percentage_val=0.)
                
                n_inputs = len(state_columns)
                n_outputs = len(control_input_columns)

                i_count += 1
                
                print('{} training with {}, system: {}'.format(i_count, algo, system_name))
                
                # linear
                X_train, y_train = convert_dict_to_np_dataset(train_dict)
                print('linear regression Data, state shape:', X_train.shape, ', action shape: ', y_train.shape)
                # Get the rotation matrix and its inverse for the input transformation:
                R, _, Rinv = np.linalg.svd(np.cov(y_train, rowvar=False))
                y_train = np.matmul(y_train, R)

                clip_y[system_name]= {
                    "min": np.amin(y_train, axis=0), 
                    "max": np.amax(y_train, axis=0)
                }

                # data process
                scaler = StandardScaler()
                scaler = scaler.fit(X_train)
                poly = PolynomialFeatures(2)
                X_train = scaler.transform(X_train)
                X_train = poly.fit_transform(X_train)
                n_inputs = X_train.shape[1]

                # # sklearn LR
                # lrg = LinearRegression()
                # o_model = lrg.fit(X_train, y_train)
                # o_actions = o_model.predict(X_train)

                # my GD LR
                model = LR_GD(n_inputs, n_outputs, X_train, y_train)
                actions = model.predict(X_train)

                # print(np.amin(o_actions, axis=0), np.amax(o_actions, axis=0))
                # print(np.amin(actions, axis=0), np.amax(actions, axis=0))
                
                # o_y_train_pred = np.matmul(o_actions, Rinv)
                y_train_pred = np.matmul(actions, Rinv)

                print(
                    # f"lr MAE for system {system_name} is :",
                    # np.mean(np.abs(o_y_train_pred.flatten() - y_train.flatten())),
                    # '\n'
                    f"my gd_lr MAE for system {system_name} is :",
                    np.mean(np.abs(y_train_pred.flatten() - y_train.flatten())),
                )

                
                # Save nn model and scaler
                model_save_path = (
                    f"{system_name}_model"
                )
                model_save_path = model_path + model_save_path
                torch.save(model.state_dict(), model_save_path)

                r_save_path = (
                    f"{system_name}_model_clipping_R"
                )
                r_save_path = model_path + r_save_path
                with open(r_save_path, "wb") as f:
                    pickle.dump(Rinv, f)

                scaler_path = (f"{system_name}_scaler")
                scaler_path = model_path + scaler_path

                with open(scaler_path, "wb") as f:
                    pickle.dump(scaler, f)
                
                

                system_name_info[system_name]= {
                    "n_inputs": n_inputs, 
                    "n_outputs": n_outputs
                }
                print(f'-- {system_name} finish training-- ')
                
            # save system_name_info
            system_name_info_path  =f"system_name_info"
            system_name_info_path = model_path + system_name_info_path
            with open(system_name_info_path, "wb") as f:
                pickle.dump(system_name_info, f)
            
            clip_y_path = model_path + 'clip_y'
            with open(clip_y_path, "wb") as f:
                pickle.dump(clip_y, f)

            print(f'-- finish training-- ')
        
        