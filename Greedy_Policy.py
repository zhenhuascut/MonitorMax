# -*- coding:utf-8 -*-
"""
Find monitor nodes with Greedy policy
"""
import networkx as nx
import matplotlib.pyplot as plt

def read_graph(graph_name):
    graph = nx.DiGraph()
    ba_graph = nx.random_graphs.barabasi_albert_graph(250,3)
    graph = ba_graph.to_directed()
    return graph

def choose_max(node_preds, pred_nodes_set, choosed_nodes):
    node_pred_num = {}
    max_pred_num = 0
    max_node = None
    for node in node_preds:
        if node not in choosed_nodes:
            preds = set(node_preds[node])
            print(preds)
            preds = preds - pred_nodes_set
            preds = preds - choosed_nodes
            print(preds)
            node_pred_num[node] = len(preds)
            if len(preds)> max_pred_num:
                max_pred_num = len(preds)
                max_node = node

    return max_node, max_pred_num

def GreedyPolicy(graph, topk=5):
    node_preds = {}
    for node in graph.nodes():
        node_preds[node] = graph.predecessors(node)

    choosed_nodes = set()
    pred_nodes_set = set()
    for i in range(topk):
        choosed_node,max_pred_num = choose_max(node_preds, pred_nodes_set, choosed_nodes)
        print(choosed_node, max_pred_num)
        choosed_nodes.add(choosed_node)
        pred_nodes = graph.predecessors(choosed_node)
        print(pred_nodes)
        pred_nodes_set = pred_nodes_set.union(set(pred_nodes))
        print pred_nodes_set
    print pred_nodes_set

    return pred_nodes_set

if __name__ == '__main__':
    graph = read_graph(graph_name="")
    monitor_nodes_set = GreedyPolicy(graph)
    nx.draw(graph, with_labels=True)
    plt.show()
    print(len(monitor_nodes_set))