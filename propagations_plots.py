import sys
sys.path = ['/home/corradom/projects/WoMG/src'] + sys.path
import datetime
import pandas as pd
import os
from womg.__main__ import womg_main
import multiprocessing
import numpy as np
import uuid

N_JOBS = 10
N_TOPICS = 10
N_DOCS = 200
N_STEPS = 100
god_node_strength = 0
int_mode = 'nmf'   
graph_path = '/home/corradom/projects/WoMG/src/womg/high-clustered-sf.nx'
assert os.path.exists(graph_path)
# graph_path = '../data/graph/barabasi/barabasi_edgelist.txt'
directed = False
nr_experiments = 10 

single_activators = (False, True)
infl_strengths = (None, .5, 1, 2)
homophily = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
virality = [0.25, 0.5, 1, 1.5, 2, 4]
# vir2text = {0.25: "Very high", 0.5: "High", 1: "Medium", 1.5: "Medium-Low", 2.: "Low", 4: "Very low"}

timestamp = datetime.datetime.now().isoformat().replace(":", "-")
results_path = 'results' + timestamp + '.csv'
simulat_path = 'sim' + timestamp + '/'

def run_one_experiment(arguments):
    seed, single_activator, infl_strength, h, v = arguments
    
    # Create output folder
    path = simulat_path + str(uuid.uuid4())
    os.mkdir(path)
    
    # Run experiments
    womg_main(path_out=path, graph_path=graph_path,
              directed=directed, int_mode=int_mode,
              numb_topics=N_TOPICS, numb_docs=N_DOCS, numb_steps=N_STEPS, 
              homophily=h, virality=v, gn_strength=god_node_strength,
              single_activator=single_activator,
              infl_strength=infl_strength,
              seed=seed)
              
    # Read results
    file_prop = path + "/Propagations0.txt"
    df = pd.read_csv(file_prop, sep=' ', names=['time', 'item', 'node'])
    item_per_node = df.groupby('node').item.nunique().values
    node_per_item = df.groupby('item').node.nunique().values
    
    # Write results
    new_row = pd.DataFrame([[single_activator, infl_strength, h, v, seed, 
        np.mean(item_per_node), np.mean(node_per_item), 
        item_per_node, node_per_item
    ]])
    with open(results_path, 'a') as f:
        new_row.to_csv(f, header=False)
        f.flush()

if __name__ == '__main__':
    os.mkdir(simulat_path)
    
    with open(results_path, 'w') as f:
        f.write('single_activator, infl_strength, homophily, virality_exp, seed, '
                'mean_item_per_node, mean_node_per_item, '
                'item_per_node, node_per_item\n')

    args = []
    for seed in range(nr_experiments):
        for single_activator in single_activators:
            for infl_strength in infl_strengths:
                for h in homophily:
                    for v in virality:
                        args.append([seed, single_activator, infl_strength, h, v])
    
    pool = multiprocessing.Pool(N_JOBS)
    pool.map(run_one_experiment, args)
