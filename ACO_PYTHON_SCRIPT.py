import pandas as pd
import numpy as np
import random
import math
import gc
from collections import defaultdict
import matplotlib.pylab as plt
import networkx as nx
from copy import deepcopy
import time

random.seed(1)
num_vertices = 50
num_ants = 20
edges = []
vertices = list(range(num_vertices))

# Initialize adjacency matrix
def init_adjacency_matrix(nodes, edges):
    adjacency_matrix = np.zeros((nodes,nodes),dtype = np.int)
    edge_probability = .0085 if nodes > 100 else 0.09
    edges_cnt = 0
    for i in range(nodes):
        for j in range(i):
            prob = random.random()
            if prob < edge_probability:
                adjacency_matrix[i,j] = 1
                edges.append((i,j))
                edges_cnt += 1
    ant_adjacency_matrix = {ant:deepcopy(adjacency_matrix) for ant in range(num_ants)}
    return ant_adjacency_matrix, edges
ant_adjacency_matrix, edges = init_adjacency_matrix(num_vertices, edges)

G=nx.Graph()
G.add_nodes_from(list(range(num_vertices)))
G.add_edges_from(edges)

# Initialize variables and constants
ant_dict = {ant: [random.choice(range(num_vertices)), 0] for ant in range(num_ants)} # ant: (curr_vertex, num_of_vertices_visited)
vertex_weights = {node:1 for node in range(num_vertices)}
tau_o = random.random() 
ant_solutions = {ant: [] for ant in range(num_ants)}
ant_finisher = None
q_o = 0.5
alpha = 0.5
psi = 0.3

#Vertex Cover Greedy Algorithm
visited = np.zeros(num_vertices)
greedy_vertex_dict = {}
greedy_edges = deepcopy(edges)

for vertex in range(num_vertices):
    greedy_vertex_dict[vertex] = sum(ant_adjacency_matrix[0][vertex])/vertex_weights[vertex]

greedy_vertex_dict_ordered = dict(sorted(greedy_vertex_dict.items(), key=lambda item: item[1], reverse=True))

greedy_count = 0
vcount = 0
for vertex in greedy_vertex_dict_ordered:
    to_remove = set()
    for edge in greedy_edges:
        if vertex in edge:
            to_remove.add(edge)
    for edge in to_remove:
        greedy_edges.remove(edge)
    greedy_count +=vertex_weights[vertex]
    vcount += 1
    if len(greedy_edges) == 0: 
        break

print(f"Vertex_greedy_count: {vcount}, Weight_count_greedy: {greedy_count}")

# Helpers
def get_available_vertices(ant, ant_adjacency_matrix):
    vertices = []
    for ind,row in enumerate(ant_adjacency_matrix[ant]):
        if sum(row) > 0:
            vertices.append(ind)
    return list(set(vertices) - set(ant_solutions[ant]))

def get_arg_max_phermone():
    arg_max = []
    max_val = -1
    for vertex, value in vertex_values.items():
        # print(value, max_val)
        if value > max_val:
            max_val = value
            arg_max = [vertex]
            
        elif value == max_val:
            arg_max.append(vertex)
            
    return arg_max
def get_psi_k(ant, ant_adjacency_matrix, vertex_i, vertex_j):
    return ant_adjacency_matrix[ant][vertex_i][vertex_j]

# Eta function
def get_eta_k(ant, ant_adj_matrix, vertex):
    total_edges = sum([get_psi_k(ant, ant_adj_matrix, vertex_i,vertex_j) for vertex_i in range(num_vertices) for vertex_j in range(num_vertices)])
    weight_vertex = vertex_weights[vertex]
    return total_edges/weight_vertex

vertex_phermones = {node:tau_o for node in range(num_vertices)}
vertex_values = {node: vertex_phermones[node]*get_eta_k(0,ant_adjacency_matrix, node)**alpha for node in range(num_vertices)}
print("Ant dict is ", ant_dict)

# Probability
def get_transition_probabilty(ant,vertex_j,vertex_i=None, q_o=q_o, available_vertices = None, ant_adjacency_matrix=ant_adjacency_matrix):
    # vertex_j is current vertex and vertex_i is adjacent vertex
    q = random.random()
    if q <=q_o or not vertex_i:
        tau_j = vertex_phermones[vertex_j]
        eta_j_k = (get_eta_k(ant, ant_adjacency_matrix, vertex_j))**alpha
        if not available_vertices:
            available_vertices = get_available_vertices(ant, ant_adjacency_matrix)
        denominator = sum([tau_j*get_eta_k(ant, ant_adjacency_matrix, vert) for vert in available_vertices])
        return tau_j*eta_j_k / denominator
    
    arg_max = get_arg_max_phermone()
    if vertex_i in arg_max:
        return 1
    return 0
        

# Local update rule
def get_local_tau_j(vertex_j):
    return (1-psi)*vertex_phermones[vertex_j] + psi*tau_o
    
# Global update rule 
def get_delta_tau_j(ant):
    total_weights = sum([vertex_weights[vertex_j] for vertex_j in ant_solutions[ant]])
    return 1/(total_weights) if total_weights > 0 else 0
    
def get_tau(ant, vertex_i, local_phermone):
    delta_tau = get_delta_tau_j(ant)
    tau = (1-get_transition_probabilty(ant,vertex_i))*(local_phermone+vertex_phermones[vertex_i]) + delta_tau # We update based on local and current global phermones
    return tau

def update_eta_k(ant, vertex, available_vertices = None, ant_adjacency_matrix=ant_adjacency_matrix):
    if not available_vertices:
        available_vertices = get_available_vertices(ant, ant_adjacency_matrix)
    # print("fnerfrewfw",sum([vertex_phermones[node]*get_eta_k(node)**alpha for node in available_vertices]))
    vertex_values[vertex] = sum([vertex_phermones[node]*get_eta_k(ant, ant_adjacency_matrix, node)**alpha for node in available_vertices])
    return 
        

def an_ant_finished(ant_adjacency_matrix):
    global ant_finisher
    ants_count = 0
    for ant in range(num_ants):
        if ants_finished(ant, ant_adjacency_matrix):
            ant_finisher = ant
            return True
    return False

# Change this function
def ants_finished(ant, ant_adjacency_matrix):
    ants_finished = True
    for row in ant_adjacency_matrix[ant]:
        if sum(row) != 0: 
            ants_finished = False
            break
    if ants_finished:
        print(f"{ant} finished")
    return ants_finished

def visit_vertex_steps(adj_vertex, num_vertices_visited, ant):
    ant_solutions[ant].append(adj_vertex)
    ant_dict[ant] = [adj_vertex, num_vertices_visited+1]
    
    # Update adjacency matrix
    ant_adjacency_matrix[ant][adj_vertex] = [0]*num_vertices
    for row in range(num_vertices):
        ant_adjacency_matrix[ant][row][adj_vertex] = 0
    return 

def start_ACO(ant_adjacency_matrix):
    local_phermone_dict = defaultdict(int)
    i = 0
    for ant, (curr_vertex, num_vertices_visited) in ant_dict.items():
        visit_vertex_steps(curr_vertex, num_vertices_visited, ant)
        
    break_now = False
    global ant_solutions
    
    while not (an_ant_finished(ant_adjacency_matrix)):
        print(i)
        i+=1
        local_phermone_dict = defaultdict(int) # vertex: phermone used to sotre local phermone status
        tic_ant = time.time()
        for ant, (curr_vertex, num_vertices_visited) in ant_dict.items():
            if num_vertices_visited == num_vertices:
                print("{ant} done")
                continue# Ant has not visited all vertices; ant is not finished
            random_prob =  random.random()
            tic1 = time.time()
            available_vertices = get_available_vertices(ant, ant_adjacency_matrix)
            # print(f"For available vertices time: {time.time() - tic1} ant: {ant}")
            if not available_vertices:
                print(f"{ant} is DONE!")
                ant_finisher = ant
            tic2 = time.time()
            random.shuffle(available_vertices)
            # print(f"Random shuffle time: {time.time() - tic1} ant: {ant}")
            tic3 = time.time()
            for adj_vertex in available_vertices:
                if adj_vertex in ant_solutions[ant]:
                    continue
                tic4 = time.time()
                prob = get_transition_probabilty(ant,curr_vertex, adj_vertex, available_vertices=available_vertices)
                # print(f"Get transition prob time: {time.time() - tic4} ant: {ant}")
                random_prob =  random.random()
                if random_prob <= prob:
                    visit_vertex_steps(adj_vertex, num_vertices_visited, ant)
                        
                    # Update local phermone
                    local_phermone_val = get_local_tau_j(adj_vertex)
                    local_phermone_dict[adj_vertex] += local_phermone_val
                    
                    # Update eta 
                    update_eta_k(ant, adj_vertex) 
                    break
            # print(f"ant: {ant} time: {time.time() - tic_ant}")
            # print(time.time() - tic3)
        for ant in range(num_ants):
        
            for vertex, local_phermone in local_phermone_dict.items():
                # Not sure if this should be local or global
                vertex_phermones[vertex] =  get_tau(ant, vertex, local_phermone) # This function uses local and global phermones for update
    return ant_solutions    


solutions_found = start_ACO(ant_adjacency_matrix)
solutions = solutions_found[ant_finisher]
print("Ant finisher is", ant_finisher)

solutions = solutions_found[ant_finisher]
print(f"Solutions: {solutions}")