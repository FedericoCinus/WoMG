# Class for random distributions definition
import random
import numpy as np
import tensorflow as tf

def set_seed(seed):
    '''
    Sets the given seed for each distribution extraction
    '''
    if seed != None:
        random.seed(seed)
        np.random.seed(seed)
        tf.set_random_seed(seed)

def f(x0, x1, gamma=1.5):
    y = np.random.rand()
    return pow(((pow(x1,gamma+1) - pow(x0,gamma+1))*y + pow(x0,gamma+1)),(1/(gamma+1)))

def random_powerlaw_vec(gamma, dimensions):
    '''
    Returns the virality vector, which is a numb_docs dimension vector of integers
    extracted from a power law distribution with exponent equal to gamma paramself.

    x = [(x1^(n+1) - x0^(n+1))*y + x0^(n+1)]^(1/(n+1))
    where y is a uniform variate, n is the distribution power, x0 and x1 define
    the range of the distribution, and x is your power-law distributed variate.
    '''
    samples = []
    for i in range(dimensions):
        samples.append(f(10E-7, 1, -gamma))
    return samples

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
