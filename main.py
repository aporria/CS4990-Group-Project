import sys

from mpi4py import MPI
import numpy as np
import networkx as nx
import scipy as sp


def main():
    #facebook_combined.txt
    np.set_printoptions(threshold=sys.maxsize)
    file = input("Enter name of file to analyze: ")
    digraph = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=int, edgetype=int)
    graph = digraph.to_undirected()
    #adj_mat = nx.convert.to_dict_of_dicts(digraph)
    #adj_mat = nx.convert.to_dict_of_lists(digraph)
    adj_mat = nx.to_numpy_matrix(digraph)
    print(adj_mat)


    def dijkstra_p():
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()

        def min_dist():
            # encontra a menor distancia no set local
            dist = float('inf')
            pos = -1
            for index, distance in enumerate(set):
                if distance < dist:
                    dist = distance
                    pos = index
            return set[pos]

        dist = {}
        prev = {}


if __name__ == '__main__':
    main()
