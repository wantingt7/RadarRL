import numpy as np
import pandas as pd
import traceback
from os import listdir
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

figure_name = "dqn vs ppo vs a2c.jpg"
smooth_value = 0      # 0 for non-smooth
logdir_dict = {
    "dqn": "log_temp/test6/test_dqn_seeds",
    "mbdqn": "log_temp/test6/mbdqn_seeds",
    "medqn": "log_temp/test6/medqn_seeds",

    "ppo": "log_temp/test6/test_ppo_seeds",
    "mbppo": "log_temp/test6/mbppo_seeds",

    "a2c": "log_temp/test6/test_a2c_seeds",
    "mba2c": "log_temp/test6/mba2c_seeds",
    "mea2c": "log_temp/test6/mea2c_seeds",
}

color_list = [
    '#C72E29',
    '#C06000',
    '#C9C400',
    '#5DC400',
    '#00C495',
    '#0079C4',
    '#9A00C4',
    '#7D7D7D',
    '#FF1DBE',
    '#704444',
]

def path_processing(logdir_dict):
    log_path_dict = {}
    for algo in logdir_dict:
        root_dir = logdir_dict[algo]
        path_list = []
        if algo.find("mb") != -1 or algo.find("me") != -1:
            for dir in listdir(root_dir):
                path_list.append(root_dir + '/' + dir + "/" + algo + "_real")
        else:
            for dir in listdir(root_dir):
                path = root_dir + '/' + dir
                path = path + "/" + listdir(path)[0]
                path_list.append(path)
        log_path_dict[algo] = path_list

    print(log_path_dict)
    return log_path_dict

def tflog2pandas(path: str) -> pd.DataFrame:
    """convert single tensorflow log file to pandas DataFrame
    Parameters
    ----------
    path : str
        path to tensorflow log file
    Returns
    -------
    pd.DataFrame
        converted dataframe
    """
    DEFAULT_SIZE_GUIDANCE = {
        "compressedHistograms": 1,
        "images": 1,
        "scalars": 0,  # 0 means load all
        "histograms": 1,
    }
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        # tags = event_acc.Tags()["scalars"]
        tags = ['test/rew'] 
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data

def many_logs2pandas(event_paths):
    all_logs = pd.DataFrame()
    for path in event_paths:
        log = tflog2pandas(path)
        if log is not None:
            if all_logs.shape[0] == 0:
                all_logs = log
            else:
                all_logs = all_logs.append(log, ignore_index=True)
    return all_logs

def std(x):
    return x.std(ddof=0)

def cal_statistics(df):
    result = df.groupby(['step'], as_index=False).agg({'value':['mean', std]})
    result.columns = ['step', 'mean', 'std']
    return result

def smooth(scalars, weight):  # Weight between 0 and 1
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        last = smoothed_val                                  # Anchor the last smoothed value
        
    return smoothed

def draw(path_dict):
    plt.figure(figsize=(10, 6), dpi=80)
    plt.xlabel("Steps", fontsize=12)
    plt.ylabel("Test Rewards", fontsize=12)

    id = 0
    algos = []
    for algo in path_dict:
        # data processing
        algos.append(algo)
        df = many_logs2pandas(path_dict[algo])
        print(df)
        sta = cal_statistics(df)
        print(sta)

        # first color: line, second color: shadow
        x = sta['step']
        mean = sta['mean']
        se = sta['std']

        color = color_list[id]
        id += 1

        # smoothed
        plt.plot(x, smooth(mean, smooth_value), color=color, lw=2)
        plt.fill_between(x, mean-se, mean+se, color=color, alpha = 0.15)

    # Lighten borders
    plt.gca().spines["top"].set_alpha(0)
    plt.gca().spines["bottom"].set_alpha(1)
    plt.gca().spines["right"].set_alpha(0)
    plt.gca().spines["left"].set_alpha(1)
    # plt.xticks(x[::6], [str(d) for d in x[::6]], fontsize=12)

    plt.title("Algorithms Comparison",fontsize=14)

    # Axis limits
    # s, e = plt.gca().get_xlim()
    plt.xlim(0, 50000)
    plt.ylim(0, 10)

    # Draw Horizontal Tick lines
    for y in np.arange(0, 10, 0.5):
        plt.hlines(y, xmin=0, xmax=5e4,colors='black',alpha=0.5,linestyles="--",lw=0.5)

    # Draw Vertical Tick lines
    for x_ in range(1000, 50000, 1000):
        plt.vlines(x_, ymin=0, ymax=10,colors='black',alpha=0.5,linestyles="--",lw=0.5)

    plt.legend(labels=algos,  loc='upper right')
    plt.savefig('./figure/' + figure_name)
    plt.show()


if __name__ == "__main__":
    log_path_dict = path_processing(logdir_dict)
    draw(log_path_dict)
    