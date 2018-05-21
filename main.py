import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from cvxpy import *
from ripl.ripl.graph import Graph
from tqdm import tqdm
from util import aspl_lower_bound

# Figure 1a)
def figure1a():
    N = 40
    degrees = range(3, 32, 2)
    all2all = []

    for degree in tqdm(degrees):
        random_graph = Graph.rrg(N, degree)
        d_star = aspl_lower_bound(degree, N)

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

        # all2all
        num_flows = N * (N - 1)
        num_links = N * degree
        # flow (start --> end) at link (u, v) identified as
        # flow_id(start --> end) * num_links + link_id(u, v)
        flow_var = Variable((num_flows, num_links))
        K = Variable()  # min flow

        objective = Maximize(K)
        constraints = []

        # Assign flow_id
        flow_id = 0
        flow_ids = {}
        for i in xrange(1, N + 1):
            for j in xrange(1, N + 1):
                if i != j:
                    flow_ids[(random_graph.get_vertex(i),
                              random_graph.get_vertex(j))] = flow_id
                    flow_id += 1

        # Assign link_id
        link_id = 0
        link_ids = {}
        for vertex in random_graph.vertices():
            for neighbor in vertex.neighbors:
                link_ids[(vertex, neighbor)] = link_id
                link_id += 1

        for flow in flow_ids:
            start, end = flow
            from_start = [
                flow_var[flow_ids[flow], link_ids[(start, neighbor)]]
                for neighbor in start.neighbors]
            constraints.append(sum(from_start) >= K)

            to_end = [flow_var[flow_ids[flow], link_ids[(neighbor, end)]]
                      for neighbor in end.neighbors]
            constraints.append(sum(to_end) >= K)

            for vertex in random_graph.vertices():
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
        print "Degree: {}".format(degree)
        print "Flows: {}".format(flow_var.value)
        print "Solution: {}".format(result)
        upper_bound = float(N * degree) / (d_star * num_flows)
        print "Upper bound: {}".format(upper_bound)

        print "Ratio: {}".format(result / upper_bound)


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
