import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from cvxpy import *
from ripl.ripl.graph import Graph
from tqdm import tqdm
from util import aspl_lower_bound

def total_flows(traffic):
    return sum([1 for src in traffic for dst in traffic[src] if src != dst])


def generate_lp(graph, N, degree, traffic):
    """Generates and solves a linear program corresponding to the max-min fair
    flow allocation through the graph. Returns the throughput of the worst-off
    flow.

    Args:
        graph (Graph)
        N (int): number of vertices in the graph
        degree (int): degree of each vertex in the graph
        traffic (dict(int: list[int])): maps each vertex to the other vertices
            it sends to

    Returns:
        float
    """
    # LP (variable for each flow, link)
    # max K
    # s.t. for each flow (start, end)
    #
    # Flow exiting start = flow entering end = K
    # sum(flow_(start, end) (start, v) for each link (start, v)) >= K
    # sum(flow_(start, end) (v, end) for each link (v, end)) >= K
    #
    # For each node v
    # Flow exiting node v = flow entering node v
    # sum(flow_(start, end) (u, v) for each link (u, v)) =
    # sum(flow_(start, end) (v, u) for each link (v, u))
    #
    # Flow at each link below capacity
    # For each link (u, v)
    # sum(flow_(start, end) (u, v) for each flow (start, end)) <= 1
    #
    # all flows positive

    # Assign flow_id
    flow_id = 0
    flow_ids = {}
    for src in traffic:
        for dst in traffic[src]:
            if src == dst:
                continue

            flow_ids[(graph.get_vertex(src),
                      graph.get_vertex(dst))] = flow_id
            flow_id += 1

    # Assign link_id
    link_id = 0
    link_ids = {}
    for vertex in graph.vertices():
        for neighbor in vertex.neighbors:
            link_ids[(vertex, neighbor)] = link_id
            link_id += 1

    num_links = N * degree
    # flow (start --> end) at link (u, v) identified as
    # flow_id(start --> end), link_id(u, v)
    flow_var = Variable((flow_id, num_links))
    K = Variable()  # min flow

    objective = Maximize(K)
    constraints = []

    for flow in flow_ids:
        start, end = flow
        from_start = [
            flow_var[flow_ids[flow], link_ids[(start, neighbor)]]
            for neighbor in start.neighbors]
        constraints.append(sum(from_start) >= K)

        to_end = [flow_var[flow_ids[flow], link_ids[(neighbor, end)]]
                  for neighbor in end.neighbors]
        constraints.append(sum(to_end) >= K)

        for vertex in graph.vertices():
            if vertex == start or vertex == end:
                continue

            flows_out = [
                    flow_var[flow_ids[flow], link_ids[(vertex, neighbor)]]
                    for neighbor in vertex.neighbors]
            flows_in = [
                    flow_var[flow_ids[flow], link_ids[(neighbor, vertex)]]
                    for neighbor in vertex.neighbors]
            constraints.append(sum(flows_in) == sum(flows_out))

    for link in link_ids:
        link_flows = [flow_var[flow_ids[flow], link_ids[link]]
                      for flow in flow_ids]
        constraints.append(sum(link_flows) <= 1)

    for flow in flow_ids:
        for link in link_ids:
            constraints.append(
                    flow_var[flow_ids[flow], link_ids[link]] >= 0)

    prob = Problem(objective, constraints)
    result = prob.solve()
    return result


# Figure 1a)
def figure1a():
    N = 40
    degrees = range(3, 32, 2)
    all2all = []
    rand5perm = []
    rand10perm = []

    for degree in tqdm(degrees):
        random_graph = Graph.rrg(N, degree)
        d_star = aspl_lower_bound(degree, N)
        #all2all_traffic = {i: range(1, N + 1) for in xrange(1, N + 1)}
        #upper_bound = float(N * degree) / (d_star * total_flows(all2all))
        #all2all_bound = generate_lp(random_graph, N, degree, all2all)
        #all2all.append(all2all_bound / upper_bound)

        rand5perm_traffic = range(1, N + 1) * 5
        np.random.shuffle(rand5perm_traffic)
        rand5perm_traffic = {rand5perm_traffic[i]:
                             [rand5perm_traffic[(i + 1) % len(rand5perm_traffic)]]
                             for i in xrange(len(rand5perm_traffic))}
        upper_bound = float(N * degree) / (d_star * total_flows(rand5perm_traffic))
        rand5perm_bound = generate_lp(random_graph, N, degree, rand5perm_traffic)
        rand5perm.append(rand5perm_bound / upper_bound)
        print "Upper bound: {}".format(upper_bound)
        print "Actual: {}".format(rand5perm_bound)
        print "Ratio: {}".format(rand5perm_bound / upper_bound)

        #rand10perm_traffic = range(1, N + 1) * 10
        #np.random.shuffle(rand10perm_traffic)
        #rand10perm_traffic = {rand10perm_traffic[i]:
        #                     [rand10perm_traffic[(i + 1) % len(rand10perm_traffic)]]
        #                     for i in xrange(len(rand10perm_traffic))}
        #upper_bound = float(N * degree) / (d_star * total_flows(rand10perm_traffic))
        #rand10perm_bound = generate_lp(random_graph, N, degree, rand10perm_traffic)
        #rand10perm.append(rand10perm_bound / upper_bound)

    plt.plot(degrees, rand5perm, label="Permutation (5 Servers per switch)")
    plt.xlabel("Network Degree")
    plt.ylabel("Throughput (Ratio to Upper-bound)")
    plt.legend()
    plt.savefig("figure1a.png")


# Figure 1b)
def figure1b():
    N = 40
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

# Figure 2b)
def figure2b():
    Ns = range(15, 200, 10)
    degree = 10
    observed = []
    lower_bound = []

    for N in Ns:
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
        print "N: {}".format(N)
        print "Observed ASLP: {}".format(empirical)
        print "ASPL lower-bound: {}".format(theoretical_bound)

    plt.plot(Ns, observed, label="Observed ASPL")
    plt.plot(Ns, lower_bound, label="ASPL lower-bound")
    plt.xlabel("Network Size")
    plt.ylabel("Path Length")
    plt.legend()
    plt.savefig("figure2b.png")

figure1a()
#figure1b()
#figure2b()
