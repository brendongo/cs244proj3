import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from ripl.ripl.graph import Graph
from tqdm import tqdm
from util import aspl_lower_bound


def total_flows(traffic):
    return sum([len(traffic[src]) for src in traffic])


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
    def permit_link(flow, link):
        flow = (flow[0].uid, flow[1].uid)
        link = (link[0].uid, link[1].uid)
        shortest_path = graph.k_shortest_paths(1, flow[0], flow[1])[0]
        # Graph is disconnected --> you're hosed
        if shortest_path is None:
            return False

        path_to_link_start = graph.k_shortest_paths(1, flow[0], link[0])[0]
        path_from_link_end = graph.k_shortest_paths(1, link[1], flow[1])[0]
        # No path through this link
        if path_to_link_start is None or path_from_link_end is None:
            return False

        # Path through link is too long
        # NOTE: These paths actually include the start vertex
        if len(shortest_path) + 3 < len(path_to_link_start) + len(path_from_link_end) + 1:
            return False
        return True

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
    flow_var = cvx.Variable((flow_id, num_links))
    K = cvx.Variable()  # min flow

    objective = cvx.Maximize(K)
    constraints = []

    for flow in flow_ids:
        start, end = flow
        from_start = [
            flow_var[flow_ids[flow], link_ids[(start, neighbor)]]
            for neighbor in start.neighbors if permit_link(flow, (start, neighbor))]
        to_start = [
            flow_var[flow_ids[flow], link_ids[(neighbor, start)]]
            for neighbor in start.neighbors if permit_link(flow, (start, neighbor))]
        if len(from_start) == 0:
            continue
        constraints.append(cvx.sum(from_start) - cvx.sum(to_start) >= K)

        to_end = [flow_var[flow_ids[flow], link_ids[(neighbor, end)]]
                  for neighbor in end.neighbors if permit_link(flow, (neighbor, end))]
        from_end = [flow_var[flow_ids[flow], link_ids[(end, neighbor)]]
                    for neighbor in end.neighbors if permit_link(flow, (neighbor, end))]
        if len(to_end) == 0:
            continue
        constraints.append(cvx.sum(to_end) - cvx.sum(from_end) >= K)

        for vertex in graph.vertices():
            if vertex == start or vertex == end:
                continue

            flows_out = [
                    flow_var[flow_ids[flow], link_ids[(vertex, neighbor)]]
                    for neighbor in vertex.neighbors if permit_link(flow, (vertex, neighbor))]
            flows_in = [
                    flow_var[flow_ids[flow], link_ids[(neighbor, vertex)]]
                    for neighbor in vertex.neighbors if permit_link(flow, (vertex, neighbor))]
            if len(flows_out) == 0:
                continue
            constraints.append(cvx.sum(flows_in) == cvx.sum(flows_out))

    for link in link_ids:
        link_flows = [flow_var[flow_ids[flow], link_ids[link]]
                      for flow in flow_ids if permit_link(flow, link)]
        if len(link_flows) == 0:
            continue
        constraints.append(cvx.sum(link_flows) <= 1)

    for flow in flow_ids:
        for link in link_ids:
            constraints.append(
                    flow_var[flow_ids[flow], link_ids[link]] >= 0)

    prob = cvx.Problem(objective, constraints)
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

        perm = range(1, N + 1) * 5
        np.random.shuffle(perm)
        rand5perm_traffic = defaultdict(list)
        for i in xrange(len(perm)):
            rand5perm_traffic[perm[i]].append(perm[(i + 1) % len(perm)])
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


# Figure 2a)
def figure2a():
    Ns = range(15, 140, 10)
    degree = 10
    all2all = []
    rand5perm = []
    rand10perm = []

    for N in tqdm(Ns):
        random_graph = Graph.rrg(N, degree)
        d_star = aspl_lower_bound(degree, N)
        #all2all_traffic = {i: range(1, N + 1) for in xrange(1, N + 1)}
        #upper_bound = float(N * degree) / (d_star * total_flows(all2all))
        #all2all_bound = generate_lp(random_graph, N, degree, all2all)
        #all2all.append(all2all_bound / upper_bound)

        perm = range(1, N + 1) * 5
        np.random.shuffle(perm)
        rand5perm_traffic = defaultdict(list)
        for i in xrange(len(perm)):
            rand5perm_traffic[perm[i]].append(perm[(i + 1) % len(perm)])
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
    plt.xlabel("Network Size")
    plt.ylabel("Throughput (Ratio to Upper-bound)")
    plt.legend()
    plt.savefig("figure2a.png")


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


def figure10():
    def expected_cross_cluster(N, n1, n2, degree):
        return n1 * n2 * degree / float(N - 1)

    def clusters(n1, n2, degree, Cs):
        upper_bounds = []
        N = n1 + n2
        total_C = N * degree
        d_star = aspl_lower_bound(degree, N)
        for C in tqdm(Cs):
            cross_cluster_capacity = C * expected_cross_cluster(N, n1, n2, degree)
            upper_bound = min(total_C / (d_star * (n1 + n2)),
                              cross_cluster_capacity * (n1 + n2) / (2 * n1 * n2))
            upper_bounds.append(upper_bound)
        return upper_bounds

    def build_cluster_graph(n1, n2, degree, cross_cluster_links):
        N = n1 + n2
        graph = Graph.rrg(N, degree)
        graph = Graph.rrg(N, degree)
        actual_cross_cluster = 0
        for i in xrange(1, n1 + 1):
            vertex = graph.get_vertex(i)
            for neighbor in vertex.neighbors:
                if neighbor.uid > n1:
                    actual_cross_cluster += 2

        while abs(actual_cross_cluster - cross_cluster_links) > 2:
            if actual_cross_cluster - cross_cluster_links < 0:
                while True:
                    v1 = graph.get_vertex(
                            np.random.randint(1, n1 + 1))
                    candidates = [n for n in v1.neighbors if n.uid <= n1]
                    if len(candidates) > 0:
                        v1_n = np.random.choice(candidates)
                        v1.remove_neighbor(v1_n)
                        v1_n.remove_neighbor(v1)
                        break

                while True:
                    v2 = graph.get_vertex(
                            np.random.randint(n1 + 1, N + 1))
                    candidates = [n for n in v2.neighbors if n1 < n.uid]
                    if len(candidates) > 0:
                        v2_n = np.random.choice(candidates)
                        v2.remove_neighbor(v2_n)
                        v2_n.remove_neighbor(v2)
                        break

                v1.add_neighbor(v2)
                v1_n.add_neighbor(v2_n)
                v2.add_neighbor(v1)
                v2_n.add_neighbor(v1_n)
                actual_cross_cluster += 2
            else:
                while True:
                    v1 = graph.get_vertex(
                            np.random.randint(1, n1 + 1))
                    candidates = [n for n in v1.neighbors if n1 < n.uid]
                    if len(candidates) > 0:
                        v1_n = np.random.choice(candidates)
                        v1.remove_neighbor(v1_n)
                        v1_n.remove_neighbor(v1)
                        break

                while True:
                    v2 = graph.get_vertex(
                            np.random.randint(n1 + 1, N + 1))
                    candidates = [n for n in v2.neighbors if n.uid <= n1]
                    if len(candidates) > 0:
                        v2_n = np.random.choice(candidates)
                        v2.remove_neighbor(v2_n)
                        v2_n.remove_neighbor(v2)
                        break

                v1.add_neighbor(v2)
                v1_n.add_neighbor(v2_n)
                v2.add_neighbor(v1)
                v2_n.add_neighbor(v1_n)
                actual_cross_cluster -= 2
        return graph

    Cs = np.arange(0.15, 1.8, 0.15)
    degree = 5
    n1 = 15
    n2 = 10
    bound_A = clusters(n1, n2, degree, Cs)
    throughput_A = []
    for C in tqdm(Cs):
        N = n1 + n2
        graph = build_cluster_graph(
                n1, n2, degree,
                C * expected_cross_cluster(N, n1, n2, degree))

        perm = range(1, N + 1)
        np.random.shuffle(perm)
        randperm_traffic = defaultdict(list)
        for i in xrange(len(perm)):
            randperm_traffic[perm[i]].append(perm[(i + 1) % len(perm)])
        throughput = generate_lp(graph, N, degree, randperm_traffic)
        throughput_A.append(throughput)

    n1 = 5
    n2 = 10
    bound_B = clusters(n1, n2, degree, Cs)
    throughput_B = []
    for C in tqdm(Cs):
        N = n1 + n2
        graph = build_cluster_graph(
                n1, n2, degree,
                C * expected_cross_cluster(N, n1, n2, degree))

        perm = range(1, N + 1)
        np.random.shuffle(perm)
        randperm_traffic = defaultdict(list)
        for i in xrange(len(perm)):
            randperm_traffic[perm[i]].append(perm[(i + 1) % len(perm)])
        throughput = generate_lp(graph, N, degree, randperm_traffic)
        throughput_B.append(throughput)
    plt.plot(Cs, bound_A, label="Bound A")
    plt.plot(Cs, throughput_A, label="Throughput A")
    plt.plot(Cs, bound_B, label="Bound A")
    plt.plot(Cs, throughput_B, label="Throughput A")
    plt.xlabel("Cross-cluster Links (Ratio to Expected Under Random Connection)")
    plt.ylabel("Normalized Throughput")
    plt.legend()
    plt.savefig("figure10a.png")


#figure1a()
#figure1b()
#figure2a()
#figure2b()
figure10()
