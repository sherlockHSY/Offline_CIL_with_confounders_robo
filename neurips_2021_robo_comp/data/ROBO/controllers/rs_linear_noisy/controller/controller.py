import pickle
import signal
from contextlib import contextmanager

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures, StandardScaler

class TimeoutException(Exception):
    pass

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

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException()

    signal.signal(signal.SIGALRM, signal_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


class controller(object):
    def __init__(self, system, d_control):
        """
        Entry point, called once when starting the controller for a newly
        initialized system.

        Input:
            system - holds the identifying system name;
                     all evaluation systems have corresponding training data;
                     you may use this name to instantiate system specific
                     controllers

            d_control  - indicates the control input dimension
        """
        # lr, lr_nn, rs_linear, rs_nn, bc
        self.algo = 'rs_linear_noisy'
        models_path = 'models_70_s_300'
        with open(f"./{models_path}/clip_y", "rb") as f:
            self.clip_y = pickle.load(f)
        
        with open(f"./{models_path}/system_name_info", "rb") as f:
            self.system_info = pickle.load(f)
        
        self.system = system
        self.d_control = d_control

        n_inputs = self.system_info[system]['n_inputs']
        n_outputs = self.system_info[system]['n_outputs']
        
        scaler_path = f"./{models_path}/{system}_scaler"
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        r_save_path = f"./{models_path}/{system}_model_clipping_R"
        with open(r_save_path, "rb") as f:
            self.Rinv = pickle.load(f)

        model_save_path = f"./{models_path}/{system}_model"
        model = LinearModel(n_inputs, n_outputs)
        model.load_state_dict(torch.load(model_save_path))
        model.eval()
        self.model = model

        # if system.endswith("beetle") or system.endswith("butterfly"):  # if polynomial cliping

            
        #     n_inputs = self.system_info[system]['n_inputs']
        #     n_outputs = self.system_info[system]['n_outputs']
            
        #     scaler_path = f"./{models_path}/{system}_scaler"
        #     with open(scaler_path, "rb") as f:
        #         self.scaler = pickle.load(f)
            
        #     r_save_path = f"./{models_path}/{system}_model_clipping_R"
        #     with open(r_save_path, "rb") as f:
        #         self.Rinv = pickle.load(f)

        #     model_save_path = f"./{models_path}/{system}_model"
        #     model = LinearModel(n_inputs, n_outputs)
        #     model.load_state_dict(torch.load(model_save_path))
        #     model.eval()
        #     self.model = model
   
            
        # elif system.endswith("bumblebee"):
        #     self.all_controllers_params = joblib.load(
        #         f"./{models_path}/linear_controllers_params.joblib"
        #     )
        #     self.controller_params = self.all_controllers_params[self.system]
        #     self.n_states = self.controller_params["n_states"]
        #     self.n_inputs = self.controller_params["n_inputs"]
        #     self.scaler = self.controller_params["scaler"]
        #     self.R = self.controller_params["R"]
        #     self.A = self.controller_params["A"]
        #     self.B = self.controller_params["B"]
        #     self.d = self.controller_params["d"]
        #     self.Binv = self.controller_params["Binv"]

    def get_input(self, state, position, target):
        """
        This function is called at each time step and expects the next
        control input to apply to the system as return value.

        Input: (all column vectors, if default wrapcontroller.py is used)
            state - vector representing the current state of the system;
                    by convention the first two entries always correspond
                    to the end effectors X and Y coordinate;
                    the state variables are in the same order as in the
                    corresponding training data for the current system
                    with name self.system
            position - vector of length two representing the X and Y
                       coordinates of the current position
            target - vector of length two representing the X and Y
                     coordinates of the next steps target position
        """ 
        model_input = np.vstack([state, target,
                               target[:2] - state[:2]]).transpose()
        # model_input = np.vstack([state]).transpose()

        scaler = self.scaler
        poly = PolynomialFeatures(2)
        model_input = scaler.transform(model_input)
        model_input = poly.fit_transform(model_input)   
        
        
        # Outputs is 1,n_us
        # my lr
        outputs = np.clip(self.model.predict(model_input),
                            self.clip_y[self.system]['min'],
                            self.clip_y[self.system]['max'])

        outputs = np.matmul(outputs, self.Rinv)
        # output of the model is going to be (1,n_u) turn it into (n_u, 1)
        return outputs.transpose()
        # if self.system.endswith("beetle") or self.system.endswith("butterfly") :
        #     model_input = np.vstack([state, target,
        #                        target[:2] - state[:2]]).transpose()
        #     # model_input = np.vstack([state]).transpose()

        #     scaler = self.scaler
        #     poly = PolynomialFeatures(2)
        #     model_input = scaler.transform(model_input)
        #     model_input = poly.fit_transform(model_input)   
            
            
        #     # Outputs is 1,n_us
        #     # my lr
        #     outputs = np.clip(self.model.predict(model_input),
        #                         self.clip_y[self.system]['min'],
        #                         self.clip_y[self.system]['max'])

        #     outputs = np.matmul(outputs, self.Rinv)
        #     # output of the model is going to be (1,n_u) turn it into (n_u, 1)
        #     return outputs.transpose()
 
        # elif self.system.endswith("bumblebee"):
        #     try:
        #         with time_limit(0.9 * 16 / 200):
        #             state = state.flatten()
        #             target = target.flatten()
        #             transformed_u = self.Binv @ (
        #                 target - self.A[:2] @ state - self.d[:2]
        #             )
        #             transformed_u = np.clip(transformed_u, -1, 1)
        #             u = self.R @ self.scaler.inverse_transform([transformed_u])[0]
        #             return u
        #     except TimeoutException:
        #         return np.zeros(self.n_inputs)

        
