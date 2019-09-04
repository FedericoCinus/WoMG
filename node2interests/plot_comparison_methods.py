
import numpy as np
import pandas as pd
import networkx as nx
from scipy.spatial import distance
from n2i.__main__ import n2i_nx_graph
from sklearn.decomposition import NMF
from tqdm import tqdm

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

def homophily(G):
    return sim_in(G) / sim_out(G, 10000)

nr_nodes = 200
G = nx.barabasi_albert_graph(nr_nodes, m=2)
for edge in G.edges():
    G[edge[0]][edge[1]]['weight'] = 1


H_val = np.arange(0, 1.05, 0.1)
dimensions = [10]
hyperparams = []
nr_experiments = 10

for d in dimensions:
    for seed in range(nr_experiments):
        for h in H_val:
            args = [d, h, seed]
            hyperparams.append(args)


def run_n2v(d, h, seed):
    '''
    Returns results list with H instead of p and q of node2vec
    '''
    wk, n, wi, ii = 80, 10, 10, 1
    d, h, seed = args
    G_emb = n2i_nx_graph(nx_graph=G, 
             dimensions=d, walk_length=wk,
             num_walks=n, window_size=wi,
             iiter=ii, p=pow(2, -12*h+6), q=pow(2, 12*h-6),
             beta=5,
             alpha_value=0.5,
             beta_value=0.5,
             prior='beta',
             seed=seed)
    for i in G.nodes:
        G.node[i]['interests'] = G_emb[i]
    si = sim_in(G)
    so = sim_out(G, 50)
    return ['n2v'] + list(args) + [si, so]


def propag_emb(G, nr_nodes, nr_topics, steps, perc_influencers, seed=42):
    '''
    generates interests with the propagation method starting from "n" nodes
    '''
    numb_influencers = int(perc_influencers/100 * nr_nodes)
    
    # INITIALIZE
    np.random.seed(seed)
    for i in list(G.nodes):
        interests = np.random.dirichlet(np.ones(nr_topics)*1./nr_topics)
        G.nodes[i]['interests'] = interests
    
    # SELECT INFLUENCERS
    # start with a random node
    influencers = [np.random.choice(G.nodes)]

    for i in range(numb_influencers):
        # calculate distances to current influencers
        sp = {i: nx.shortest_path(G, i) for i in influencers}
        distances = np.array([list(len(sp[j][i]) for j in influencers) for i in G.nodes()])
        # select the node fartest from all influencers
        influencers.append(distances.min(axis=1).argmax()) 
      
    # PROPAGATION STEP
    for c in range(steps):
        i = np.random.choice(nr_nodes)
        interests_i = G.nodes[i]['interests']
        #print(G.nodes[0]['interests'])
        lr = 0.5 if i in influencers else 0.01
        #lr = 0.1
        for j in list(G.neighbors(i)):
            if j in influencers:
                continue
            interests_j = G.nodes[j]['interests']
            interests_j += interests_i * lr
            interests_j /= interests_j.sum()
            G.nodes[j]['interests'] = interests_j
            #print(interests_j)
        

    # preparing matrix
    prop_emb = [G.nodes[_]['interests'] for _ in range(nr_nodes)]
    return np.array(prop_emb)


# In[23]:


def run_clpa(d, h, seed):
    '''
    Returns list of results with H param instead of steps and percentange of influencers
    '''
    d, h, seed = args
    G_emb =  propag_emb(G, nr_nodes, nr_topics=d, 
                        steps=int(pow(2, 7*h+3)), perc_influencers=pow(2, 7*h-8),
                        seed=seed+int(1000*(h)))
    for i in G.nodes:
        G.node[i]['interests'] = G_emb[i]
    si = sim_in(G)
    so = sim_out(G)
    return ['clpa'] + list(args) + [si, so]


# ## ---------------- NMF ----------------

# In[25]:


def overlap_generator(G):
    """
    Function to generate a neighbourhood overlap matrix (second-order proximity matrix).
    :param G: Graph object.
    :return laps: Overlap matrix.
    """
    #print("Second order proximity calculation.\n")
    degrees = nx.degree(G)
    sets = {node:set(G.neighbors(node)) for node in nx.nodes(G)}
    laps = np.array([[float(len(sets[node_1].intersection(sets[node_2])))/(float(degrees[node_1]*degrees[node_2])**0.5) if node_1 != node_2 else 0.0 for node_1 in nx.nodes(G)] for node_2 in nx.nodes(G)],dtype = np.float64)
    return laps


# In[26]:


def nmf_emb(G, dimensions, seed, eta=0.5, beta=0.):
    A = nx.to_numpy_matrix(G)
    eta = eta
    beta = beta
    S_0 = overlap_generator(G)
    R = np.random.rand(G.number_of_nodes(), G.number_of_nodes())
    
    S = eta*S_0 + A + beta*R
    model = NMF(n_components=dimensions, init='random', random_state=seed)
    W = model.fit_transform(S)
    H = model.components_
    return W


def run_nmf(d, h, seed):
    d, h, seed = args
    G_emb =  nmf_emb(G, dimensions=d,  
                        eta=2, beta=16-16*h,
                        seed=seed)
    for i in G.nodes:
        G.node[i]['interests'] = G_emb[i]
    si = sim_in(G)
    so = sim_out(G)
    return ['nmf'] + list(args) + [si, so]


with open("comparison-methods.csv", "w") as f:
    f.write('method,num_dim,H,seed,s_in,s_out\n')
    for args in tqdm(hyperparams):
        for run in (run_clpa, run_n2v, run_nmf):
            result = run(args)
            f.write(','.join([str(r) for r in result]) + '\n')
            f.flush()
            
print(pd.read_csv("comparison-methods.csv"))
