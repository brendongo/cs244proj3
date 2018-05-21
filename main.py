import numpy as np
import matplotlib.pyplot as plt
from graph import Graph
from collections import Counter
from tqdm import tqdm

N_VERTICES = 600
k = 64
np.random.seed(123)
graph = Graph.rrg(N_VERTICES, 10)

permutation = range(N_VERTICES)
np.random.shuffle(permutation)

ksp_counts = Counter()
ecmp_counts = Counter()
ecmp64_counts = Counter()
for vertex in graph._vertices.itervalues():
    for neighbor in vertex.neighbors:
        if vertex.uid < neighbor.uid:
            ksp_counts[(vertex, neighbor)] = 0
            ecmp_counts[(vertex, neighbor)] = 0
            ecmp64_counts[(vertex, neighbor)] = 0
        else:
            ksp_counts[(neighbor, vertex)] = 0
            ecmp_counts[(neighbor, vertex)] = 0
            ecmp64_counts[(neighbor, vertex)] = 0

for i in tqdm(xrange(N_VERTICES - 1)):
    shortest_paths = graph.k_shortest_paths(
            k, permutation[i], permutation[i + 1])
    for path in shortest_paths[:8]:
        for j in xrange(len(path) - 1):
            edge = (path[j], path[j + 1])
            if path[j].uid > path[j + 1].uid:
                edge = (path[j + 1], path[j])
            ksp_counts[edge] += 1

    for path in shortest_paths[:8]:
        if len(path) == len(shortest_paths[0]):
            for j in xrange(len(path) - 1):
                edge = (path[j], path[j + 1])
                if path[j].uid > path[j + 1].uid:
                    edge = (path[j + 1], path[j])
                ecmp_counts[edge] += 1

    for path in shortest_paths:
        if len(path) == len(shortest_paths[0]):
            for j in xrange(len(path) - 1):
                edge = (path[j], path[j + 1])
                if path[j].uid > path[j + 1].uid:
                    edge = (path[j + 1], path[j])
                ecmp64_counts[edge] += 1

ksp_counts = sorted(ksp_counts.values())
ecmp_counts = sorted(ecmp_counts.values())
ecmp64_counts = sorted(ecmp64_counts.values())
plt.plot(range(len(ksp_counts)), ksp_counts, label="8 shortest paths")
plt.plot(range(len(ecmp_counts)), ecmp_counts, label="8 ecmp")
plt.plot(range(len(ecmp64_counts)), ecmp64_counts, label="64 ecmp")
plt.legend()
plt.show()
