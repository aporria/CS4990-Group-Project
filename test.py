import sys
from mpi4py import MPI
import numpy as np
import networkx as nx
import pprint as pp
import matplotlib.pyplot as plt


def main():
    file = 'facebook_combined.txt'
    graph = nx.read_edgelist(file, create_using=nx.Graph(), nodetype=int, edgetype=int)
    graph.to_undirected()

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
                    value = dist[current_min_node] + 1
                else:
                    value = dist[current_min_node]
                if value < dist[neighbor]:
                    dist[neighbor] = value
                    prev[neighbor] = current_min_node

            q.remove(current_min_node)

        return dist

    """
    counter = 0
    for i in dijkstra(graph, 0):
        print(str(counter) + '\t' + str(i))
        counter = counter + 1
    
    dist_sum = 0
    for i in dijkstra(graph, 4038):
        if i != 1e7:
            dist_sum += i
        else:
            dist_sum += 0
    
    # orig_dist = len(dijkstra(graph, 0))
    # print(orig_dist)
    dijkstra(graph, 0).remove(1e7)
    new_dist = len(dijkstra(graph, 0))
    print(new_dist)
    # new_n = orig_dist - new_dist

    dist = dijkstra(graph, 4038)
    i = len(dist)
    dist = [value for value in dist if value != 1e7]
    j = len(dist)
    k = i - j
    new_n = i - k
    print(new_n)

    cc = (new_n - 1) / dist_sum
    print(cc)
    """

    # nx_cc = nx.closeness_centrality(graph, u=686)
    # print('NX:  ', nx_cc)

    def clo_cen(graph, src) -> float:
        dist = dijkstra(graph, src)
        final_dist = [value for value in dist if value != 1e7]
        dist_sum = np.sum(final_dist)
        cc = (len(final_dist) - 1) / dist_sum
        return cc

    # for i in range(len(graph)):
    #   print("CC", i, " ", clo_cen(graph, i))

    comm = MPI.COMM_WORLD
    p = comm.Get_size()
    rank = comm.Get_rank()
    local_n = len(graph.nodes) // p
    count = 0
    for i in range(p):
        if rank == i:
            for node in range(count + local_n):
                print(clo_cen(graph, node))
            count += local_n


if __name__ == '__main__':
    main()
