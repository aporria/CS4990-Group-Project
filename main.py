import sys

from mpi4py import MPI
import numpy as np
import networkx as nx
import scipy as scipy
import queue 

distances = []
local_vertices = []
dist = {}


def main():
    #facebook_combined.txt
    np.set_printoptions(threshold=sys.maxsize)
  #  file = input("Enter name of file to analyze: ")
    file = "facebook_combined.txt"
    digraph = nx.read_edgelist(file, create_using=nx.DiGraph(), nodetype=int, edgetype=int)
    graph = digraph.to_undirected()
    #adj_mat = nx.convert.to_dict_of_dicts(digraph)
    #adj_mat = nx.convert.to_dict_of_lists(digraph)
    adj_mat = nx.to_numpy_matrix(digraph)
  #  print(adj_mat)
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    source = 4037
    nodelist = list(graph.nodes)
    nodelist.remove(source)
    #if rank == 0:
           # distances[source] == 0
    count = 0
    for i in nodelist:
        i = 1E7
        # count += 1
        # print(count," " ,i)
    visited = {}
    queue = []

    for j in list(graph.nodes):
        queue.append(j)
        # if visited
        # queue.pop(visited[index])
    # while not queue # empty queue
    #while not queue.empty:
        # find minimum distance  


if __name__ == '__main__':
    main()
    
