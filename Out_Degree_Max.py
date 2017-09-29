# -*- coding:utf-8 -*-
"""
Find monitor nodes with Max_Out_Degree policy
"""
import networkx as nx
import matplotlib.pyplot as plt

def read_graph(graph_name):
    graph = nx.DiGraph()
    ba_graph = nx.random_graphs.barabasi_albert_graph(50,3)
    graph = ba_graph.to_directed()
    return graph

def OutDegreeMax(graph, topk=5):
    out_degrees = []
    for node in graph.nodes():
        out_degrees.append((node, graph.out_degree(node)))
    print(out_degrees)
    out_degrees_sorted = sorted(out_degrees, key=lambda x:x[1], reverse=True)
    print(out_degrees_sorted)

    topk_nodes = out_degrees_sorted[:topk]
    print(topk_nodes)

    pred_nodes = []
    for tnode in topk_nodes:
        node = tnode[0]
        predecesors = graph.predecessors(node)
        print(predecesors)
        pred_nodes.extend(predecesors)
    print pred_nodes
    pred_nodes_set = set(pred_nodes)
    print pred_nodes_set
    return pred_nodes_set

if __name__ == '__main__':
    graph = read_graph(graph_name="")
    monitor_nodes_set = OutDegreeMax(graph)
    nx.draw(graph, with_labels=True)
    plt.show()
    print(len(monitor_nodes_set))
