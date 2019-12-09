import sys
import datetime
import pandas as pd
import os
from womg_core.__main__ import womg_main
import multiprocessing.pool
import numpy as np
import uuid
import shutil

# N_JOBS = 4
N_TOPICS = 10
N_DOCS = 3553 # Number of items is the same in reduced and not reduced Digg
N_STEPS = 1000
god_node_strength = 0
int_mode = 'nmf'   
# graph_path = '/home/fcinus/.local/lib/python3.6/site-packages/womgdata/graph/lesmiserables/lesmiserables_edgelist.txt'
graph_path = '/home/fcinus/.local/lib/python3.6/site-packages/womgdata/graph/digg/graph_filtered.csv'

#graph_path = '/home/corradom/projects/WoMG/src/womg/high-clustered-sf.nx'
assert os.path.exists(graph_path)
# graph_path = '../data/graph/barabasi/barabasi_edgelist.txt'
directed = True
nr_experiments = 4 

single_activators = (True, )  # Digg is not single activator
infl_strengths = [None]
homophily = [0, 0.25, 0.5, 0.75, 1]
virality = [2 ** i for i in range(0, 25, 2)]

timestamp = datetime.datetime.now().isoformat().replace(":", "-")
results_path = 'digg-by-int-results' + timestamp + '.csv'
simulat_path = 'digg-by-int-sim' + timestamp + '/'

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
    if df.empty:
    	print('\n Parameters h=', h, ' v=',v , ' infl_strength=', infl_strength, ' single_Activator=', single_activator, ' provided no cascade! \n')
    item_per_node = df.groupby('node').item.nunique().values
    node_per_item = df.groupby('item').node.nunique().values
    time_per_item = df.groupby('item').time.max().values + 1
    
    # Write results
    new_row = pd.DataFrame([[single_activator, infl_strength, h, v, seed, 
        np.mean(item_per_node), np.mean(node_per_item), np.mean(time_per_item),
        item_per_node, node_per_item, time_per_item
    ]])
    #shutil.rmtree(path, ignore_errors=True)
    with open(results_path, 'a') as f:
        new_row.to_csv(f, header=False)
        f.flush()

if __name__ == '__main__':
    os.mkdir(simulat_path)
    
    with open(results_path, 'w') as f:
        f.write('single_activator,infl_strength,homophily,virality_exp,seed,'
                'mean_item_per_node,mean_node_per_item,mean_max_time,'
                'item_per_node,node_per_item,time_per_item\n')

    args = []
    for seed in range(nr_experiments):
        for single_activator in single_activators:
            for infl_strength in infl_strengths:
                for h in homophily:
                    for v in virality:
                        # args.append([seed, single_activator, infl_strength, h, v])
                        arg = [seed, single_activator, infl_strength, h, v]
                        print(arg)
                        run_one_experiment(arg)
    
    # pool = multiprocessing.pool.ThreadPool(N_JOBS)
    # pool.map(run_one_experiment, args)

