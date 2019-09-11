import sys
sys.path = ['/home/corradom/projects/WoMG/src'] + sys.path
import pathlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from womg.__main__ import womg_main


for single_activator in (False, True):
    for infl_strength in (None, .5, 1, 2):
        god_label = "single-act" if single_activator else "god-node"
        infl_label = "byinterests" if infl_strength is None else f"byinfl-{infl_strength}"
        path_out = f'sim-{god_label}-prop-{infl_label}'
        try:
            os.mkdir(path_out)
        except FileExistsError:
            pass
        
        print("Experiment with: ", single_activator, infl_strength)
        topics = [10]
        docs = [50]
        steps = [100] 
        homophily = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        virality = [0.25, 0.5, 1, 1.5, 2, 4]
        vir2text = {0.25: "Very high", 0.5: "High", 1: "Medium", 1.5: "Medium-Low", 2.: "Low", 4: "Very low"}
        god_node_strength = [0] 
        int_mode = 'nmf'              
        # graph_path = '../data/graph/barabasi/barabasi_edgelist.txt'
        graph_path = '/home/corradom/projects/WoMG/src/womg/high-clustered-sf.nx'
        #graph_path = None
        directed = False
        
        #influence_strength = [0] 

        args_list = []

        nr_experiments = 10 

        for t in topics:
            for d in docs:
                for s in steps:
                    for h in homophily:
                        for v in virality:
                            for g in god_node_strength:
                                for seed in range(nr_experiments):
                                    args = [t, d, s, h, v, g, seed]
                                    args_list.append(args)

        def experiment(args_list):
            for args in tqdm(args_list):
                t, d, s, h, v, g, seed = args
                womg_main(path_out=path_out, graph_path=graph_path,
                          directed=directed, int_mode=int_mode,
                          numb_topics=t, numb_docs=d, numb_steps=s, 
                          homophily=h, virality=v, gn_strength=g,
                          single_activator=single_activator,
                          infl_strength=infl_strength,
                          seed=42*seed+(d+t)*8)

        experiment(args_list)

        ## -------- analysis --------

        file_prop = path_out + ("Propagations"+str(0)+".txt")
        file_topic = path_out + ("Topics_descript"+str(0)+".txt")
        df = pd.read_csv(file_prop, sep=' ', names=['time', 'item', 'node'])


        def statistics(path):
            result = []
            for _ in range(len(args_list)):
                file_prop = path + str("Propagations"+str(_)+".txt")
                file_topic = path + str("Topics_descript"+str(_)+".txt")
                df = pd.read_csv(file_prop, sep=' ', names=['time', 'item', 'node'])
                mean_items_act = round(df.groupby('item').node.nunique().mean(), 2)
                mean_users_act = round(df.groupby('node').item.nunique().mean(), 2)
                #print(mean_items_act)
                result.append(args_list[_]+[mean_items_act, mean_users_act,])
            return result

        stat = statistics(path_out)
        
        df = pd.DataFrame(stat, columns=['t', 'd', 's', 'H', 'virality_exp', 'g', 'seed', 'cascade size', 'node activation'])
        df['virality'] = df.virality_exp.apply(vir2text.__getitem__)
        df.to_csv(f'plots/{god_label}-prop-{infl_label}.csv')
        
        # average items activations
        plt.tight_layout()
        sns.set_context("paper", font_scale=1.7)
        plt.figure(figsize=(7,5))
        sns.pointplot(data=df[df.H < 1], x='H', y='cascade size', hue='virality')
        plt.savefig(f'plots/{god_label}-prop-{infl_label}_left.pdf')
    #     plt.show()

        plt.tight_layout()
        sns.set_context("paper", font_scale=1.7)
        plt.figure(figsize=(7,5))
        sns.pointplot(data=df, x='H', y='node activation', hue='virality')
        plt.savefig(f'plots/{god_label}-prop-{infl_label}_right.pdf')
    #     plt.show()
