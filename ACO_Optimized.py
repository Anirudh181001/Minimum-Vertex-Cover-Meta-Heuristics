# from operator import invert
from collections import defaultdict
import numpy as np
import random as rd
import copy
from aco_helper import *
import networkx as nx
import matplotlib.pyplot as plt



#Initialize variables
def initialize(num_vertices, num_ants):
    adj_list, adj_mat, edges, G = make_graph_matrices(num_vertices) #Adjacency list of the n-hypercube graph
    return num_vertices, num_ants, adj_list, adj_mat, edges, G


def select_choice_vertex(ant, adj_list, phermone_dict, weight_dict, alpha, q_0, possible_vertices):
    probs = calc_prob(ant, adj_list, phermone_dict, weight_dict, alpha, q_0, possible_vertices)
    random_num = rd.random()
    cumsum_probs = dict(zip(np.cumsum(np.multiply(list(probs.keys()), [len(i) for i in probs.values()])), list(probs.values())))
    for prob in cumsum_probs:
        if random_num < prob:
            return rd.choice(cumsum_probs[prob])
        
def find_possible_vertices(ant, adj_list, no_visit_vertices):
    possible_vertices = set(adj_list) - set(ant.history_vertices)
    possible_vertices -= no_visit_vertices
    return list(possible_vertices)


def edges_covered(ant_edges):
    for ant,val in ant_edges:
        if not val:
            return True
        
        
def no_edge_vertices(adj_mat):
    ans = set()
    for vertex,row in enumerate(adj_mat):
        if sum(row) == 0:
            ans.add(vertex)
    return ans
    
def run_ants_on_hypercube_random_colors_optimized(n, num_ants): #  returns ((ant.number, path_length), iter)
    alpha = 2
    q_0 = 0.5
    num_vertices, num_ants, adj_list, adj_mat, edges, G = initialize(n, num_ants)
    ant_edges = {i:copy.deepcopy(edges) for i in range(num_ants)}
    pheromone_dict = {i:0.1 for i in range(num_vertices)}
    weight_dict = {i:1 for i in range(num_vertices)}
    evaporation_rate = 0.001
    iter = 0
    breaker = False
    no_visit_vertices = no_edge_vertices(adj_mat)
    
    #initialize ants 
    ants_list = [Ant(i) for i in range(num_ants)]
    return_ants_list = []
    #The fun begins
    while len(return_ants_list) != num_ants:
        print("iteration: ", iter)
        if breaker:
            break
        
        local_pheromone_dict = defaultdict(float)
        for ant_number in range(num_ants):
            curr_ant = ants_list[ant_number]
            if curr_ant in return_ants_list:
                continue
            if not ant_edges[ant_number]:
                return_ants_list.append(curr_ant)
                continue
            possible_vertices = find_possible_vertices(curr_ant, adj_list, no_visit_vertices)
            if len(possible_vertices) == 0: 
                print(f"Ant: {ant_number} is out of vertices to go to")
                continue

            else:
                choice_vertex = select_choice_vertex(curr_ant, adj_list, pheromone_dict, weight_dict, alpha, q_0, possible_vertices)
               
            curr_ant.add_to_visited(choice_vertex)
            local_pheromone_dict[choice_vertex] += 1/weight_dict[choice_vertex]
            ant_edges[ant_number] = [edge for edge in ant_edges[ant_number] if choice_vertex not in edge]
    
        for vertex in local_pheromone_dict:
            pheromone_dict[vertex] = (1-evaporation_rate)*pheromone_dict[vertex] + local_pheromone_dict[vertex]
        iter += 1
    return return_ants_list, G

ants_list, G = run_ants_on_hypercube_random_colors_optimized(100,30)
print(min([len(ant.history_vertices) for ant in ants_list]))