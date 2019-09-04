import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
from n2i.__main__ import n2i_nx_graph
from scipy.spatial import distance

# similarity between connected nodes
def sim_in(G):
    sims = []
    for i in G.nodes:
        for j in list(G.neighbors(i)):
            sims.append(1 - distance.cosine(G.nodes[i]['interests'], G.nodes[j]['interests']))
    return np.mean(sims)

def select_notedge(G):
    v1 = np.random.choice(G.nodes())
    v2 = np.random.choice(G.nodes())

    while (v1,v2) in G.edges or v1==v2:
        v1 = np.random.choice(G.nodes())
        v2 = np.random.choice(G.nodes())
    return v1, v2
#     n = nx.number_of_nodes(G)
#     while True:
#         a, b = np.random.randint(0, n, size=2)
#         if (a, b) not in G.edges:
#             return a, b

# similarity between disconnected nodes
def sim_out(G, samples=5000):
    sims_out = []
    for c in range(samples):
        i, j = select_notedge(G)
        sims_out.append(1 - distance.cosine(G.nodes[i]['interests'], G.nodes[j]['interests']))
    return np.mean(sims_out)

G = nx.barabasi_albert_graph(200, m=2)
print("G density:", nx.density(G))

for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1
    
p_val = [10 ** i for i in range(-2, 3)]
q_val = [10 ** i for i in range(-2, 3)]

dimensions=[10]
walk_length=[80]
num_walks=[10]
window_size=[10]
iiter=[1]

args_list = []

nr_experiments = 10

for d in dimensions:
    for wk in walk_length:
        for n in num_walks:
            for wi in window_size:
                for ii in iiter:
                    for p in p_val:
                        for q in q_val:
                            for seed in range(nr_experiments):
                                args = [d, wk, n, wi, ii, p, q, seed]
                                args_list.append(args)
                                
def run_experiment(*args):
    d, wk, n, wi, ii, p, q, seed = args
    G_emb = n2i_nx_graph(nx_graph=G, 
             dimensions=d, walk_length=wk,
             num_walks=n, window_size=wi,
             iiter=ii, p=p, q=q,
             beta=5,
             alpha_value=0.5,
             beta_value=0.5,
             prior='beta',
             seed=seed+int(1000*(q+p)))
    for i in G.nodes:
        G.node[i]['interests'] = G_emb[i]
    si = sim_in(G)
    so = sim_out(G)
    return args + [si, so]
    
result = list(map(lambda x: run_experiment(*x), tqdm(args_list)))
df = pd.DataFrame(result, columns=['d', 'wk', 'n', 'wi', 'ii', 'p', 'q', 'seed', 'si', 'so'])
df.to_csv("node2vec-heatmap.csv")
