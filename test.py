import sys
from mpi4py import MPI
import numpy as np
import networkx as nx
import pprint as pp
import matplotlib.pyplot as plt


def main():
    # np.set_printoptions(threshold=sys.maxsize)
    file = 'facebook_combined.txt'
    graph = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=int, edgetype=int)
    # adj_mat = nx.convert.to_dict_of_dicts(digraph)
    # adj_mat = nx.convert.to_dict_of_lists(digraph)
    adj_mat = nx.to_numpy_matrix(graph)

    # adj_mat = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes()))
    # print(adj_mat)
    # nodelist = list(graph.nodes)

    def dijkstra(g, src) -> list:
        dist = [len(g)]
        prev = [len(g)]
        q = []
        max_value = 1e7
        for node in g.nodes:
            dist.append(max_value)
            prev.append(-1)
            q.append(node)
        dist[src] = 0

        size = len(q)
        while q:
            current_min_node = None
            for node in q:
                if current_min_node is None:
                    current_min_node = node
                elif dist[node] < dist[current_min_node]:
                    current_min_node = node

            neighbors = g.neighbors(current_min_node)
            for neighbor in neighbors:
                if g.has_edge(current_min_node, neighbor):
                    tentative_val = dist[current_min_node] + 1
                else:
                    tentative_val = dist[current_min_node]
                if tentative_val < dist[neighbor]:
                    dist[neighbor] = tentative_val
                    prev[neighbor] = current_min_node

            q.remove(current_min_node)

        return dist

    # print(dijkstra(graph, 0))
    # pp.pprint(dijkstra(graph, 0))
    """
    counter = 0
    for i in dijkstra(graph, 0):
        print(str(counter) + '\t' + str(i))
        counter = counter + 1
    """
    dist_sum = 0
    for i in dijkstra(graph, 4038):
        if i != 1e7:
            dist_sum += i
        else:
            dist_sum += 0
    """
    orig_dist = len(dijkstra(graph, 0))
    print(orig_dist)
    dijkstra(graph, 0).remove(1e7)
    new_dist = len(dijkstra(graph, 0))
    print(new_dist)
    new_n = orig_dist - new_dist
    """

    dist = dijkstra(graph, 4038)
    i = len(dist)
    dist = [value for value in dist if value != 1e7]
    j = len(dist)
    k = i - j
    new_n = i - k
    print(new_n)

    cc = (new_n - 1) / dist_sum
    print(cc)

    nx_cc = nx.closeness_centrality(graph, u=4037)
    print(nx_cc)

    # print(graph.has_edge(0, 347))
    nx.draw(graph, with_labels=True)
    plt.show()


if __name__ == '__main__':
    main()
