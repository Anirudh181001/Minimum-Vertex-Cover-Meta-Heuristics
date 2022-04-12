import os
from collections import defaultdict
import matplotlib.pyplot as plt
from statistics import mean

# with open("Minimum_Vertex_Cover_Results.txt", 'w') as myfile:
#     pass

# monte_carlo_iterations = 20
# command = " & ".join(["jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.enabled=True --ExecutePreprocessor.timeout=300 minimum_vertex_cover_all_algorithms.ipynb"]*monte_carlo_iterations)
# os.system(f'cmd /k "{command}" ')

myfile = open("Minimum_Vertex_Cover_Results.txt", 'r')
file_content = myfile.readlines()

meta_heuristic_methods = defaultdict(list)

for line in file_content:
    line_content = line.split(',')
    for words in line_content:
        meta_heuristic_method, value = words.split(':')
        meta_heuristic_methods[meta_heuristic_method.strip()].append(int(value.strip()))

mean_monte_carlo = {i: mean(meta_heuristic_methods[i]) for i in meta_heuristic_methods}
print(mean_monte_carlo)
        
greedy_line = plt.plot(meta_heuristic_methods['Greedy'], label="Greedy")
unweighted_genetic_line = plt.plot(meta_heuristic_methods['Unweighted Genetic Algorithm'], label="Unweighted Genetic Algoritm")
weighted_genetic_line = plt.plot(meta_heuristic_methods['Weighted Genetic Algorithm'], label = "Weighted Genetic Algorithm")
aco_concurrent_line = plt.plot(meta_heuristic_methods['ACO Concurrent'], label="ACO Concurrent")
aco_optimized_line = plt.plot(meta_heuristic_methods['ACO Optimized'], label="ACO Optimized")

plt.legend()
plt.show()
