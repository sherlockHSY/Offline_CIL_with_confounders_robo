
import matplotlib.pyplot as plt
import pickle

import numpy as np

import pandas as pd
import seaborn as sns
import matplotlib
# matplotlib.rcParams['text.usetex'] = True
matplotlib.rc('font',family='serif', serif=['Palatino'])
sns.set(font='serif', font_scale=1.0)
sns.set_style("white", {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor': 'white',
        'lines.markeredgewidth': 1})

def setup_plot():
    fig = plt.figure()
    ax = plt.axes()
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')

    ax.set_ylim(20,130)

    ax.tick_params(axis = 'both', which = 'major', labelsize = 15)
    ax.tick_params(axis = 'both', which = 'minor', labelsize = 15)
    ax.tick_params(direction='in')

algo_arr = ['linear_noisy', 'rs_linear_noisy', 'my_1']
x_arr = [30,40,50,60,70]
# seeds_arr = [10,200,300]
seeds_arr = [100,20,10,200,300]

algo_result = {
    'linear_noisy': {
        'loss_mean': [],
        'beetle_loss_mean': [],
        'bumblebee_loss_mean': [],
        'butterfly_loss_mean': [],
        'loss_std': [],
        'beetle_loss_std': [],
        'bumblebee_loss_std': [],
        'butterfly_loss_std': []
    }, 
    'rs_linear_noisy': {
        'loss_mean': [],
        'beetle_loss_mean': [],
        'bumblebee_loss_mean': [],
        'butterfly_loss_mean': [],
        'loss_std': [],
        'beetle_loss_std': [],
        'bumblebee_loss_std': [],
        'butterfly_loss_std': []
    }, 
    'my_1': {
        'loss_mean': [],
        'beetle_loss_mean': [],
        'bumblebee_loss_mean': [],
        'butterfly_loss_mean': [],
        'loss_std': [],
        'beetle_loss_std': [],
        'bumblebee_loss_std': [],
        'butterfly_loss_std': []    
    }
}

for algo in algo_arr:
    for num_traj in x_arr:
        # print(f"results algo:{algo}, num_traj {num_traj}")
        loss_arr = []
        beetle_loss_arr = []
        bumblebee_loss_arr = []
        butterfly_loss_arr = []
        for seed in seeds_arr:
            res = np.load(f"neurips_2021_robo_comp/results/{algo}_{num_traj}_s_{seed}.npy",allow_pickle=True)
            res = res[0]
            loss = res['loss']
            results = res['results']
            
            beetle_count = 0
            beetle_loss = 0
            bumblebee_count = 0
            bumblebee_loss = 0
            butterfly_count = 0
            butterfly_loss = 0
            for i in results:
                robotname = i['robotname']
                loss_i = i['loss'][0]
                if robotname.endswith("beetle"):
                    beetle_loss += loss_i
                    beetle_count += 1
                if robotname.endswith("bumblebee"):
                    bumblebee_loss += loss_i
                    bumblebee_count += 1
                if robotname.endswith("butterfly"):
                    butterfly_loss += loss_i
                    butterfly_count += 1
            
            beetle_loss = beetle_loss/beetle_count
            bumblebee_loss = bumblebee_loss/bumblebee_count
            butterfly_loss = butterfly_loss/butterfly_count
            loss_arr.append(loss)
            beetle_loss_arr.append(beetle_loss)
            bumblebee_loss_arr.append(bumblebee_loss)
            butterfly_loss_arr.append(butterfly_loss)
        
        algo_result[algo]['loss_mean'].append(np.mean(loss_arr))
        algo_result[algo]['beetle_loss_mean'].append(np.mean(beetle_loss_arr))
        algo_result[algo]['bumblebee_loss_mean'].append(np.mean(bumblebee_loss_arr))
        algo_result[algo]['butterfly_loss_mean'].append(np.mean(butterfly_loss_arr))

        algo_result[algo]['loss_std'].append(np.std(loss_arr))
        algo_result[algo]['beetle_loss_std'].append(np.std(beetle_loss_arr))
        algo_result[algo]['bumblebee_loss_std'].append(np.std(bumblebee_loss_arr))
        algo_result[algo]['butterfly_loss_std'].append(np.std(butterfly_loss_arr))


    print(
            f"{algo} total loss mean is :", algo_result[algo]['loss_mean'], " std is :", algo_result[algo]['loss_std'],
            '\n'
            f"beetle loss mean is :", algo_result[algo]['beetle_loss_mean'], " std is :", algo_result[algo]['beetle_loss_std'],
            '\n'
            f"butterfly loss mean is :", algo_result[algo]['butterfly_loss_mean']," std is :", algo_result[algo]['butterfly_loss_std'],
            '\n'
            f"bumblebee loss mean is :", algo_result[algo]['bumblebee_loss_mean']," std is :", algo_result[algo]['bumblebee_loss_std'],
            '\n'
        )




setup_plot()
plt.xlabel('Num of Expert Trajectories', fontsize=15)   
plt.ylabel('Average Loss', fontsize=15)  
plt.title("Confounded RobotEnv", fontsize=15)
result = algo_result['linear_noisy']
plt.plot(np.arange(30, 71, 10), result['loss_mean'], label="BC", color="#009988")
plt.fill_between(np.arange(30, 71, 10),
                 result['loss_mean'] - (result['loss_std'] / np.sqrt(len(seeds_arr))),
                 result['loss_mean'] + (result['loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#009988",
                 alpha = 0.1)

result = algo_result['rs_linear_noisy']
plt.plot(np.arange(30, 71, 10), result['loss_mean'], label="ResiduIL", color="#F79646")
plt.fill_between(np.arange(30, 71, 10),
                 result['loss_mean'] - (result['loss_std'] / np.sqrt(len(seeds_arr))),
                 result['loss_mean'] + (result['loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#F79646",
                 alpha = 0.1)

result = algo_result['my_1']
plt.plot(np.arange(30, 71, 10), result['loss_mean'], label="Our Method", color="#049fc8")
plt.fill_between(np.arange(30, 71, 10),
                 result['loss_mean'] - (result['loss_std'] / np.sqrt(len(seeds_arr))),
                 result['loss_mean'] + (result['loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#049fc8",
                 alpha = 0.1)

plt.legend(ncol=1, loc="upper right", fontsize=15)
plt.show()  

# ---------------------------------------------------------------------------------------------------

# beetle
setup_plot()
plt.xlabel('Num of Expert Trajectories', fontsize=15)  
plt.ylabel('Average Loss', fontsize=15)  
plt.title("Confounded Beetle System", fontsize=15)
result = algo_result['linear_noisy']
plt.plot(np.arange(30, 71, 10), result['beetle_loss_mean'], label="BC", color="#009988")
plt.fill_between(np.arange(30, 71, 10),
                 result['beetle_loss_mean'] - (result['beetle_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['beetle_loss_mean'] + (result['beetle_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#009988",
                 alpha = 0.1)

result = algo_result['rs_linear_noisy']
plt.plot(np.arange(30, 71, 10), result['beetle_loss_mean'], label="ResiduIL", color="#F79646")
plt.fill_between(np.arange(30, 71, 10),
                 result['beetle_loss_mean'] - (result['beetle_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['beetle_loss_mean'] + (result['beetle_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#F79646",
                 alpha = 0.1)

result = algo_result['my_1']
plt.plot(np.arange(30, 71, 10), result['beetle_loss_mean'], label="Our Method", color="#049fc8")
plt.fill_between(np.arange(30, 71, 10),
                 result['beetle_loss_mean'] - (result['beetle_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['beetle_loss_mean'] + (result['beetle_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#049fc8",
                 alpha = 0.1)
plt.legend(ncol=1, loc="upper right", fontsize=15)
plt.show()  

# butterfly
setup_plot()
plt.xlabel('Num of Expert Trajectories', fontsize=15)  
plt.ylabel('Average Loss', fontsize=15)  
plt.title("Confounded Butterfly System", fontsize=15)
result = algo_result['linear_noisy']
plt.plot(np.arange(30, 71, 10), result['butterfly_loss_mean'], label="BC", color="#009988")
plt.fill_between(np.arange(30, 71, 10),
                 result['butterfly_loss_mean'] - (result['butterfly_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['butterfly_loss_mean'] + (result['butterfly_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#009988",
                 alpha = 0.1)

result = algo_result['rs_linear_noisy']
plt.plot(np.arange(30, 71, 10), result['butterfly_loss_mean'], label="ResiduIL", color="#F79646")
plt.fill_between(np.arange(30, 71, 10),
                 result['butterfly_loss_mean'] - (result['butterfly_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['butterfly_loss_mean'] + (result['butterfly_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#F79646",
                 alpha = 0.1)

result = algo_result['my_1']
plt.plot(np.arange(30, 71, 10), result['butterfly_loss_mean'], label="Our Method", color="#049fc8")
plt.fill_between(np.arange(30, 71, 10),
                 result['butterfly_loss_mean'] - (result['butterfly_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['butterfly_loss_mean'] + (result['butterfly_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#049fc8",
                 alpha = 0.1)
plt.legend(ncol=1, loc="upper right", fontsize=15)
plt.show() 

# bumblebee
setup_plot()
plt.xlabel('Num of Expert Trajectories', fontsize=15)  
plt.ylabel('Average Loss', fontsize=15) 
plt.title("Confounded Bumblebee System", fontsize=15)
result = algo_result['linear_noisy']
plt.plot(np.arange(30, 71, 10), result['bumblebee_loss_mean'], label="BC", color="#009988")
plt.fill_between(np.arange(30, 71, 10),
                 result['bumblebee_loss_mean'] - (result['bumblebee_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['bumblebee_loss_mean'] + (result['bumblebee_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#009988",
                 alpha = 0.1)

result = algo_result['rs_linear_noisy']
plt.plot(np.arange(30, 71, 10), result['bumblebee_loss_mean'], label="ResiduIL", color="#F79646")
plt.fill_between(np.arange(30, 71, 10),
                 result['bumblebee_loss_mean'] - (result['bumblebee_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['bumblebee_loss_mean'] + (result['bumblebee_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#F79646",
                 alpha = 0.1)

result = algo_result['my_1']
plt.plot(np.arange(30, 71, 10), result['bumblebee_loss_mean'], label="Our Method", color="#049fc8")
plt.fill_between(np.arange(30, 71, 10),
                 result['bumblebee_loss_mean'] - (result['bumblebee_loss_std'] / np.sqrt(len(seeds_arr))),
                 result['bumblebee_loss_mean'] + (result['bumblebee_loss_std'] / np.sqrt(len(seeds_arr))),
                 color = "#049fc8",
                 alpha = 0.1)
plt.legend(ncol=1, loc="upper right", fontsize=15)
plt.show() 







