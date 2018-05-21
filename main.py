import numpy as np
import matplotlib.pyplot as plt
from ripl.ripl.graph import Graph
from collections import Counter
from tqdm import tqdm
from util import aspl_lower_bound

# Figure 1b)
N = 40
flow = range(1, N + 1)
np.random.shuffle(flow)
degrees = range(3, 32)
observed = []
lower_bound = []

for degree in tqdm(degrees):
    theoretical_bound = aspl_lower_bound(degree, N)
    random_graph = Graph.rrg(N, degree)

    empirical = 0
    for i in xrange(1, N + 1):
        for j in xrange(i + 1, N + 1):
            path = random_graph.k_shortest_paths(1, i, j)[0]
            empirical += len(path) - 1
    empirical = float(empirical) / (N * (N - 1) / 2.)
    observed.append(empirical)
    lower_bound.append(theoretical_bound)
    print "Degree: {}".format(degree)
    print "Observed ASLP: {}".format(empirical)
    print "ASPL lower-bound: {}".format(theoretical_bound)

plt.plot(degrees, observed, label="Observed ASPL")
plt.plot(degrees, lower_bound, label="ASPL lower-bound")
plt.xlabel("Network Degree")
plt.ylabel("Path Length")
plt.legend()
plt.savefig("figure1b.png")