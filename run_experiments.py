import os
import random

"""
algo_list: list
    list of the name of program: test_dqn, mbdqn, medqn, ..., etc.
    all share the same set of seeds, generated randomly at beginning
seeds_num: int
    number of seeds to run, i.e., number of experiments for each algo
jammer_type: int
    type of the jammer policy in the environment
root_dir: str
    the root directory of all the log data
------------
log_path_dict: dict
    dictionary of the log path of each algo: {algo: [list of log paths]}, will be saved in log_paths.txt
"""

algo_list = ["test_dqn"]
seeds_num = 4
jammer_type = 2
root_dir = "log_temp/test8"

def run_experiments(): 
    # generate seeds
    seed_list = [638, 894, 2360, 2595]
    # for _ in range(seeds_num):
    #     seed_list.append(random.randint(1, 3000))

    # run algos
    for seed in seed_list:
        for algo in algo_list:
            file_name = algo + ".py"
            logdir = root_dir + "/" + algo + "_type" + str(jammer_type) + "_seeds"
            path = logdir + "/" + algo + "_" + str(seed)
            os.system('python3 {} --jammer_policy_type {} --seed {} --logdir {}'.format(file_name, str(jammer_type), str(seed), path))

if __name__ == "__main__":
    run_experiments()
