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

    dict_dataset_sorted = sorted(dict_dataset.items(), key=lambda x: x[0])
    dict_dataset = dict(dict_dataset_sorted)

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
    

# 残差 线性回归版
def My_ResiduIL_linear(X_past, X, y, n_inputs, n_outputs, lr=1e-3, e_1=1e4, e_2=1e4,
                    f_norm_penalty=1e-3, bc_reg=5e-2, batch_size=128, wd=1e-3):
    

    sample_size = len(X)
    
    loss_func = nn.MSELoss()

    pi = LinearModel(n_inputs, n_outputs).to(device)
    optimizer_pi = optim.Adam(pi.parameters(), lr=lr)
    
    f = LinearModel(n_inputs, n_outputs).to(device)
    optimizer_f = optim.Adam(f.parameters(), lr=lr)

    X = torch.from_numpy(X).float().to(device)
    y = torch.from_numpy(y).float().to(device)
    X_past = torch.from_numpy(X_past).float().to(device)
    
    time_slice = int(sample_size/2)
    # time_slice = sample_size

    # confoundered
    X_c = X[:time_slice,:]
    X_past_c = X_past[:time_slice,:]
    y_c = y[:time_slice,:]
    # unconfoundered
    X_uc = X[time_slice:,:]
    X_past_uc = X_past[time_slice:,:]
    y_uc = y[time_slice:,:]

    for step in range(int(e_1)):
        # idx = np.random.choice(len(X), batch_size)
        pi_inputs = X_c
        f_inputs = X_past_c
        targets = y_c

        optimizer_pi.zero_grad()
        preds = pi(pi_inputs)
        pred_residuals = f(f_inputs)
        loss = torch.mean(2 * (targets - preds) * pred_residuals)
        loss = loss + bc_reg * torch.mean(torch.square(targets - preds))

        loss.backward()
        optimizer_pi.step()
    
        optimizer_f.zero_grad()
        preds = pi(pi_inputs)
        pred_residuals = f(f_inputs)
        loss_2 = -torch.mean(2 * (targets - preds) * pred_residuals - pred_residuals * pred_residuals)
        loss_2 = loss_2 + f_norm_penalty * torch.linalg.norm(pred_residuals)
        loss_2.backward()
        optimizer_f.step()

    for e in range(int(e_2)):
        # train
        X_batch = X_uc
        y_batch = y_uc
        X_past_batch = X_past_uc

        # Compute and print loss
        pred_y = pi.forward(X_batch)
        loss = loss_func(pred_y, y_batch)
        optimizer_pi.zero_grad()
        loss.backward()
        optimizer_pi.step() 
        
    return pi


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
        help="choose seed",
    )
    parser.add_argument(
        "-c",
        "--cuda",
        type=int,
        default=0,
        help="whether to use gpu, 0 mean use cpu, 1 mean use gpu",
    )

    parser.add_argument(
        "-l",
        "--lr",
        type=float,
        default=3e-4,
        help="learning rate",
    )
    parser.add_argument(
        "-e",
        "--epochs_1",
        type=float,
        default=2e5,
        help="工具变量回归",
    )
    parser.add_argument(
        "-p",
        "--epochs_2",
        type=float,
        default=1e4,
        help="线性回归",
    )
    input_args = sys.argv[1:]
    args = parser.parse_args(input_args)
    # 参数 -------------------------------------------------------
    seed = args.seed
    num_traj = args.traj_num
    lr = args.lr
    epochs_1 = args.epochs_1
    epochs_2 = args.epochs_2
    f_norm_penalty = 1e-3
    bc_reg = 5e-2
    batch_size = 512
    optimizer = 'Adam'
    # -------------------------------------------------------------
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

    # training
    num_arr = [30,40,50,60,70]
    for i_t in num_arr:
        num_traj = i_t
        # algo
        algo = 'my_1'

        # noisy data
        data_path = f'output/robo_train_noisy_random_0.5_{num_traj}/'
        systems_path = f"output/robo_train_noisy_random_0.5_{num_traj}/systems"

        
        model_path = f"data/ROBO/controllers/{algo}/controller/models_{num_traj}_s_{seed}/"
        if not os.path.exists(model_path):
            print(model_path)
            os.makedirs(model_path)
        
        
        with open(systems_path, "r") as f:
            systems_list = [i.strip() for i in f.readlines()]

        system_name_info = {}
        clip_y = {}
        i_count = 0
        scalers = {}
        
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

            X_train, X_past, y_train = rs_convert_dict_to_np_dataset(train_dict)

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

            X_past = StandardScaler().fit(X_past).transform(X_past)
            X_past = poly.fit_transform(X_past)

            # # sklearn LR
            # lrg = LinearRegression()
            # o_model = lrg.fit(X_train, y_train)
            # o_actions = o_model.predict(X_train)

            # my GD LR + ResiduIL linear
            # model = LR_GD(n_inputs, n_outputs, X_train, y_train)

            model= My_ResiduIL_linear(X_past, X_train, y_train, n_inputs, n_outputs, 
                                    lr=lr, 
                                    e_1=epochs_1,
                                    e_2=epochs_2,
                                    f_norm_penalty=f_norm_penalty, 
                                    bc_reg=bc_reg, 
                                    batch_size=batch_size)

            actions = model.predict(X_train)
            y_train_pred = np.matmul(actions, Rinv)

            
            print(np.amin(y_train, axis=0), np.amax(y_train, axis=0))
            print(np.amin(y_train_pred, axis=0), np.amax(y_train_pred, axis=0))
            
            print(
                f"my lr MAE for system {system_name} is :",
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
        
        print('---------------finish training--------------------------')
        
        
