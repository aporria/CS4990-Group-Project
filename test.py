import sys
from mpi4py import MPI
import numpy as np
import networkx as nx
import pprint as pp
import matplotlib.pyplot as plt
from datetime import datetime


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

    # nx_cc = nx.closeness_centrality(graph)
    # pp.pprint(nx_cc)

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
    local_n = int(len(graph.nodes) / p)
    count = 0

    def mpi_clo():
        if rank == 0:
            for node in range(0, 1009):
                print(node, "\t", clo_cen(graph, node), "\t", rank)
            print('\n')
        elif rank == 1:
            for node in range(1010, 2019):
                print(node, "\t", clo_cen(graph, node), "\t", rank)
            print('\n')
        elif rank == 2:
            for node in range(2020, 3029):
                print(node, "\t", clo_cen(graph, node), "\t", rank)
            print('\n')
        elif rank == 3:
            for node in range(3030, 4039):
                print(node, "\t", clo_cen(graph, node), "\t", rank)
            print('\n')
        # print(rank)
        comm.Barrier()

    cc_list = []

    def test():
        cc = 0
        if rank == 0:
            for node in range(0, local_n):
                cc = clo_cen(graph, node)
                print(node, "\t", cc, "\t", rank)
                with open("output.txt", "a") as o:
                    o.write(str(cc))
                    o.write('\n')
                cc_list.append(cc)
            print('\n')
        elif rank != 0:
            for node in range(local_n * rank, local_n * (rank + 1)):
                cc = clo_cen(graph, node)
                print(node, "\t", cc, "\t", rank)
                with open("output.txt", "a") as o:
                    o.write(str(cc))
                    o.write('\n')
                cc_list.append(cc)
            print('\n')
        comm.Barrier()

    def average(list) -> float:
        list_sum = sum(list)
        avg = list_sum / len(graph.nodes)
        return avg

    start = datetime.now()
    test()

    end = datetime.now()
    runtime = end - start
    print("time:\t", runtime, "\trank: ", rank)

    def print_output():
        # output avg
        list_avg = average(cc_list)
        print('Average of all closeness centralities: ', list_avg)

        # output top five nodes
        sorted(cc_list, reverse=True)
        top_five = cc_list[:5]
        print('Top five nodes: ')
        for i in top_five:
            print(i)

    print_output()


if __name__ == '__main__':
    main()
