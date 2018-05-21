import copy
import numpy as np
from collections import deque


class Graph(object):
    @classmethod
    def rrg(cls, num_vertices, vertex_degree):
        """Returns a Graph with the given number of vertices, generated randomly
        according to the Jellyfish algorithm.

        Args:
            num_vertices (int)
            vertex_degree (int)

        Returns:
            Graph
        """
        def unconnected_vertices(vertices):
            candidates = []
            for i in xrange(len(vertices)):
                for j in xrange(i + 1, len(vertices)):
                    if vertices[i].degree < vertex_degree and \
                            vertices[j].degree < vertex_degree and \
                            vertices[i] not in vertices[j].neighbors:
                        candidates.append((vertices[i], vertices[j]))
            return candidates

        vertices = [Vertex(i) for i in xrange(num_vertices)]
        while True:
            candidates = unconnected_vertices(vertices)
            if not candidates:
                break
            selected_edge = candidates[np.random.randint(len(candidates))]
            selected_edge[0].add_neighbor(selected_edge[1])
            selected_edge[1].add_neighbor(selected_edge[0])

        free_vertices = \
            [v for v in vertices if v.degree <= vertex_degree - 2]
        edges = [(v, neighbor) for v in vertices
                 for neighbor in v.neighbors if v.uid < neighbor.uid]

        while len(free_vertices) > 0:
            selected_vertex = np.random.choice(free_vertices)
            selected_edge = edges[np.random.randint(len(edges))]
            if selected_vertex != selected_edge[0] and \
                    selected_vertex != selected_edge[1]:
                selected_edge[0].remove_neighbor(selected_edge[1])
                selected_edge[1].remove_neighbor(selected_edge[0])
                edges.remove(selected_edge)

                # Add the new edges
                selected_vertex.add_neighbor(selected_edge[0])
                selected_edge[0].add_neighbor(selected_vertex)
                selected_vertex.add_neighbor(selected_edge[1])
                selected_edge[1].add_neighbor(selected_vertex)

                if selected_vertex.uid < selected_edge[0].uid:
                    edges.append((selected_vertex, selected_edge[0]))
                    edges.append((selected_vertex, selected_edge[1]))
                elif selected_vertex.uid < selected_edge[1].uid:
                    edges.append((selected_edge[0], selected_vertex))
                    edges.append((selected_vertex, selected_edge[1]))
                else:
                    edges.append((selected_edge[0], selected_vertex))
                    edges.append((selected_edge[1], selected_vertex))

                # Update free vertices
                if selected_vertex.degree >= vertex_degree - 1:
                    free_vertices.remove(selected_vertex)
        return Graph(vertices)

    def __init__(self, vertices):
        """Creates a graph with these vertices.

        Args:
            vertices (list[Vertex])
        """
        self._vertices = dict((v.uid, v) for v in vertices)

    def k_shortest_paths(self, k, start, end):
        """Returns the k shortest paths between start and end.

        Args:
            k (int)
            start (int): uid of start Vertex
            end (int): uid of end Vertex

        Returns:
            list[list[Vertex]]
        """
        def bfs(start, end):
            visited = set([start])
            bfs_queue = deque([[start]])

            while len(bfs_queue) > 0:
                path_prefix = bfs_queue.popleft()
                last_vertex = path_prefix[-1]
                if last_vertex == end:
                    return path_prefix

                for neighbor in last_vertex.neighbors:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        path_copy = copy.copy(path_prefix)
                        path_copy.append(neighbor)
                        bfs_queue.append(path_copy)

            return None

        if k == 1:
            return bfs(self._vertices[start], self._vertices[end])

        # Yen's shortest path algorithm, pseudocode from Wikipedia
        A = [bfs(self._vertices[start], self._vertices[end])]
        B = []
        for _ in xrange(k - 1):
            for i in xrange(len(A[-1]) - 1):
                spur_node = A[-1][i]
                root_path = A[-1][:i + 1]

                edges_removed = []
                vertices_removed = []
                for path in A:
                    if len(path) > i + 1 and root_path == path[:i + 1]:
                        edges_removed.append((path[i], path[i + 1]))
                        path[i].remove_neighbor(path[i + 1])
                        path[i + 1].remove_neighbor(path[i])

                for vertex in root_path[:-1]:
                    vertices_removed.append(vertex)
                    self._remove_vertex(vertex)

                spur_path = bfs(spur_node, self._vertices[end])
                if spur_path is not None:
                    total_path = root_path[:-1] + spur_path
                    for path in B:
                        if tuple(total_path) == tuple(path):
                            break
                    else:
                        B.append(total_path)

                for vertex in vertices_removed:
                    self._vertices[vertex.uid] = vertex
                    for neighbor in vertex.neighbors:
                        neighbor.add_neighbor(vertex)

                for edge in edges_removed:
                    edge[0].add_neighbor(edge[1])
                    edge[1].add_neighbor(edge[0])

            if len(B) == 0:
                break
            B.sort(key=lambda x: len(x))
            A.append(B[0])
            B.pop(0)
        return A

    def _remove_vertex(self, vertex):
        for neighbor in vertex.neighbors:
            neighbor.remove_neighbor(vertex)
        del self._vertices[vertex.uid]

    def __str__(self):
        s = ""
        for v in self._vertices.itervalues():
            s += "{} --> {}\n".format(v, sorted(v.neighbors))
        return s
    __repr__ = __str__


class Vertex(object):
    def __init__(self, uid):
        """
        Args:
            uid (int): unique identifier
        """
        self.uid = uid
        self._neighbors = set()

    def add_neighbor(self, neighbor):
        self._neighbors.add(neighbor)

    def remove_neighbor(self, neighbor):
        if neighbor not in self.neighbors:
            print "Neighbor {} not found when removing from {}".format(
                    neighbor, self)
            return
        self._neighbors.remove(neighbor)

    @property
    def neighbors(self):
        return self._neighbors

    @property
    def degree(self):
        return len(self._neighbors)

    def __str__(self):
        return "Vertex({})".format(self.uid)
    __repr__ = __str__
