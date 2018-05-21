import numpy as np
from ripl.routing import Routing
from collections import defaultdict


class JellyRouting(Routing):
    def __init__(self, topo, path_filter, k=8):
        self._topo = topo
        self._k = k
        self._path_filter = path_filter
        self._paths = defaultdict(dict)

    def get_route(self, src, dst, pkt):
        if src == dst:
            return [src]
        src = self._topo.id_gen(name=src).dpid
        dst = self._topo.id_gen(name=dst).dpid
        if src not in self._paths or dst not in self._paths[src]:
            graph = self._topo._graph
            #paths = graph.k_shortest_paths(1, src, dst)
            #paths = [self._topo.id_gen(dpid=x.uid).name_str() for path in paths for x in path]
            #self._paths[src][dst] = paths

            paths = self._path_filter(
                    graph.k_shortest_paths(self._k, src, dst))
            paths = [
                [self._topo.id_gen(dpid=x.uid).name_str() for x in path] for path in paths]
            self._paths[src][dst] = paths
        return self._paths[src][dst][
                np.random.randint(len(self._paths[src][dst]))]
