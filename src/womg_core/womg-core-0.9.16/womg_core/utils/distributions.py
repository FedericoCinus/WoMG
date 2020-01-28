# Class for random distributions definition
import random
import numpy as np

def set_seed(seed):
    '''
    Sets the given seed for each distribution extraction
    '''
    if seed != None:
        random.seed(seed)
        np.random.seed(seed)

# todo
def random_powerlaw_vec(gamma, dimensions):
    '''
    Returns the virality vector, which is a numb_docs dimension vector of integers
    extracted from a power law distribution with exponent equal to gamma paramself.

    We random extract from uniform distribution and we invert the previous eq
    '''
    return list(np.random.pareto(gamma, dimensions) + 1)

def random_initial_active_set(self, max_active_perc=0.5):
    '''Returns list of active nodes;
       max_active_perc is the maximum perc (of N) of active nodes on an item
       if random_config arg is False same initial configuration will be built
    '''
    numb_nodes = int(self.network_model.info['numb_nodes'])
    nodes_list = [i for i in range(numb_nodes)]
    max_active = int(max_active_perc * numb_nodes)
    active_nodes = set(random.sample(nodes_list, random.randint(0, max_active)))

    return active_nodes
