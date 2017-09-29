# -*- coding:utf-8 -*-
import networkx as nx
import matplotlib.pyplot as plt
BA_directed = nx.random_graphs.barabasi_albert_graph(130, 2)
nx.draw(BA_directed)
plt.show()

BA_directed.clear()
nx.draw(BA_directed)
plt.show()