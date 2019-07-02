# distutils: language=c++
#cython: boundscheck=False, wraparound=False, nonecheck=False

import random
cimport cython

from libcpp.list cimport list as cpplist
from libcpp.vector cimport vector as cppvector
from libcpp.unordered_set cimport unordered_set as cppset
from libcpp.map cimport map as cppmap
from libcpp.unordered_map cimport unordered_map
from libcpp.pair cimport pair
from libc.math cimport floor
from cython.operator import dereference as deref, predecrement as decr, address as addr, preincrement as inc


ctypedef cppset[int]* int_set_ptr
ctypedef cppvector[int]* int_vec_ptr

cdef class AliasTable():
    cdef public cppvector[double] q
    cdef public cppvector[int] J
    
    def __cinit__(self, int size):
        self.q.resize(size)
        self.J.resize(size)

cdef class Graph():
    
    cdef bint is_directed
    cdef bint is_weighted
    cdef double p
    cdef double q
    cdef bint verbose
    
    cdef cppvector[int] nodes
    cdef object edges
    cdef object edge_weights
    
    cdef unordered_map[int, int_vec_ptr]* neighbors
    cdef unordered_map[int, int_set_ptr]* predecessors
    
    cdef public object alias_nodes
    cdef public object alias_edges
    
    def __cinit__(self, nx_G, is_directed, is_weighted, p, q, verbose):
        
        cdef int_vec_ptr node_neighs_list
        cdef cppvector[int] nodes = cppvector[int](len((nx_G.nodes())))
        
        cdef int_set_ptr neighs_map
        
        if not self.is_directed:
            nx_G = nx_G.to_undirected()
        
        self.neighbors = new unordered_map[int, int_vec_ptr]()
        
        for i, node in enumerate(sorted(nx_G.nodes())):
            nodes[i] = node
            node_neighs = sorted(nx_G.neighbors(node))
            node_neighs_list = new cppvector[int](len(node_neighs))
            for i, neigh in enumerate(node_neighs):
                deref(node_neighs_list)[i] = neigh
            deref(self.neighbors)[node] = node_neighs_list
        
        self.nodes = nodes
        
        self.predecessors = new unordered_map[int, int_set_ptr]()
        for node in nx_G.nodes():
            neighs_map = new cppset[int]()
            if not is_directed:
                node_neighs = nx_G.neighbors(node)
            else:
                node_neighs = nx_G.predecessors(node)
            for neigh in node_neighs:
                neighs_map.insert(neigh)
            deref(self.predecessors)[node] = neighs_map
            
    
    def __init__(self, nx_G, is_directed, is_weighted, p, q, verbose):
        self.is_directed = is_directed
        self.is_weighted = is_weighted
        self.p = p
        self.q = q
        self.verbose = verbose
        
        if not self.is_directed:
            nx_G = nx_G.to_undirected()
        
        self.edges = set(nx_G.edges)
        self.edge_weights = {}
        for edge in nx_G.edges():
            self.edge_weights[edge] = nx_G[edge[0]][edge[1]]['weight']
            
    def format_graph(self):
        pass
    
    def number_of_nodes(self):
        return len(self.nodes)
    
    cpdef node_neighbors(self, int n):
        return deref(deref(self.neighbors)[n])

    cpdef node2vec_walk(self, unsigned int walk_length, int start_node):
        '''
        Simulate a random walk starting from start node.
        '''
        cdef cppvector[int]* cur_nbrs
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        cdef cpplist[int] walk
        cdef int next
        cdef int prev
        cdef cpplist[int].iterator it
        cdef AliasTable at
        
        walk.push_back(start_node) #= [start_node]

        while walk.size() < walk_length:
            cur = walk.back()
            cur_nbrs = deref(self.neighbors)[cur]
            if deref(cur_nbrs).size() > 0:
                if walk.size() == 1:
                    at = alias_nodes[cur]
                    next = deref(cur_nbrs)[alias_draw(at.J, at.q)]
                    walk.push_back(next)
                else:
                    #prev = walk[-2]
                    it = walk.end()
                    decr(it)
                    decr(it)
                    prev = deref(it)
                    
                    at = alias_edges[(prev, cur)]
                    next = deref(cur_nbrs)[alias_draw(at.J, at.q)]
                    walk.push_back(next)
            else:
                break

        return list(walk)

    cpdef simulate_walks(self, int num_walks, unsigned int walk_length):
        '''
        Repeatedly simulate random walks from each node.
        '''
        walks = []
        nodes = list(self.nodes)
        cdef int node

        range_num_walks = range(num_walks)

        for walk_iter in range_num_walks:
            #print(str(walk_iter+1), '/', str(num_walks))
            random.shuffle(nodes)
            for node in nodes:
                walks.append(self.node2vec_walk(walk_length, node))

        return walks
    
    cdef AliasTable get_alias_edge(self, object src, object dst):
        '''
        Get the alias edge setup lists for a given edge.
        '''
        cdef double p = self.p
        cdef double q = self.q
        cdef object edges = self.edges

        cdef cppvector[double] probs
        cdef double norm_const = 0.
        cdef bint is_weighted = self.is_weighted
        cdef double weight
        
        cdef int_vec_ptr dst_nbrs = deref(self.neighbors)[dst]
        cdef int_set_ptr src_predecessors = deref(self.predecessors)[src]
        
        probs.resize(deref(dst_nbrs).size())
        cdef int i = 0
        
        for dst_nbr in deref(dst_nbrs):
            
            if is_weighted:
                weight = self.edge_weigths[(dst, dst_nbr)]
            else:
                weight = 1.
            
            if dst_nbr == src:
                probs[i] = weight/p
            
            # elif (dst_nbr, src) in edges:
            elif deref(src_predecessors).find(dst_nbr) != deref(src_predecessors).end():
                probs[i] = weight
            else:
                probs[i] = weight/q
                
            i += 1
        
        for v in probs:
            norm_const += v
        i = 0
        for v in probs:
            probs[i] = v/norm_const
            i += 1
        
        return alias_setup(&probs)

    def preprocess_transition_probs(self):
        '''
        Preprocessing of transition probabilities for guiding the random walks.
        '''
        cdef bint is_directed = self.is_directed
        cdef bint is_weighted = self.is_weighted

        alias_nodes = {}
        
        cdef cppvector[double] probs
        cdef double norm_const
        cdef double weight
        
        cdef int i
        cdef int n
        
        for node in self.nodes:
            
            n = deref(self.neighbors)[node].size()
            probs.resize(n)
            
            if not is_weighted:
                
                n = deref(deref(self.neighbors)[node]).size()
                for i in range(n):
                    probs[i] = 1./n
            
            else:
                norm_const = 0.
                i = 0

                for nbr in deref(deref(self.neighbors)[node]): #G.neighbors(node):
                    if is_weighted:
                        weight = self.edge_weights[(node, nbr)]
                    else:
                        weight = 1.
                    probs[i] =  weight
                for v in probs:
                    norm_const += v
                i = 0
                for v in probs:
                    probs[i] = v/norm_const
                    i += 1
            
            alias_nodes[node] = alias_setup(&probs)

        alias_edges = {}

        edges = self.edges
        
        for edge in edges:
            alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
            if not is_directed:
                alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

cdef AliasTable alias_setup(cppvector[double]* probs):
    '''
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    '''
    cdef int K = deref(probs).size()
    
    cdef AliasTable at = AliasTable(K)

    cdef cpplist[int] smaller
    cdef cpplist[int] larger
    
    cdef int small
    cdef int large
    
    cdef double temp
    
    cdef int kk = 0
    
    for prob in deref(probs):
        temp = K*prob
        at.q[kk] = temp
        if temp < 1.0:
            smaller.push_back(kk)
        else:
            larger.push_back(kk)
        kk += 1

    while smaller.size() > 0 and larger.size() > 0:
        small = smaller.back()
        smaller.pop_back()
        large = larger.back()
        larger.pop_back()

        at.J[small] = large
        at.q[large] = at.q[large] + at.q[small] - 1.0
        if at.q[large] < 1.0:
            smaller.push_back(large)
        else:
            larger.push_back(large)
    
    return at

@cython.cdivision(True)
cpdef int alias_draw(cppvector[int]& J, cppvector[double]& q):
    '''
    Draw sample from a non-uniform discrete distribution using alias sampling.
    '''
    cdef int K = J.size()
    cdef float r = random.random()
    cdef int kk = int(floor(r*K))
    
    r = random.random()
    if r < q[kk]:
        return kk
    else:
        return J[kk]
