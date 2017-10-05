#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/10/3 23:52        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：蜂群算法获取最大化监视节点 √ ━━━━━☆*°☆*°
"""
import networkx as nx
import random
import Out_Degree_Max
import Greedy_Policy

def read_graph(graph_name):
	graph = nx.DiGraph()
	ba_graph = nx.random_graphs.barabasi_albert_graph(250, 3)
	graph = ba_graph.to_directed()
	return graph

# 从父节点中随机替换节点，ex_node为被topk中被选中的将要被替换的节点，batch_nodes为某一批topk节点
# x_random是除了ex_node之外的topk节点,max_try_times为随机替换父节点最大尝试次数
def choose_node(node_preds, ex_node, batch_nodes, x_random, max_try_times = 100):
	ex_node_pred = node_preds[ex_node]
	# print ex_node_pred
	pred_sum = 0
	# 计算ex_node父节点的所有出度和
	for node in ex_node_pred:
		#
		each_node_pre = node_preds[node]
		pred_sum += len(each_node_pre)
	# 计算每个父节点被选择的概率
	each_p = [len(node_preds[node])/float(pred_sum) for node in ex_node_pred]
	# 移除ex_node节点，准备替换
	batch_nodes.remove(ex_node)
	pred_nodes = []
	for node in batch_nodes:
		pred_nodes.extend(node_preds[node])
	pred_nodes = set(pred_nodes)
	# print pred_nodes
	# 原批次节点最大监视节点集合
	origin_node_set = pred_nodes.union(set(node_preds[ex_node]))
	find_flag = 0
	while max_try_times > 0 and find_flag == 0:
		max_try_times -= 1
		p_sum, bx_node = 0.0, 0
		choose_random = random.random()
		# 轮盘赌选择节点进行替换
		for p in each_p:
			sums = p_sum + p
			if choose_random >= p_sum and choose_random < sums:
				bx_node = each_p.index(p)
				# print bx_node
				# bx_node为选择的节点的编号
				break
			p_sum += p
		new_node_set = pred_nodes.union(set(node_preds[bx_node]))
		# print len(origin_node_set), len(new_node_set)
		if len(new_node_set) > len(origin_node_set):
			batch_nodes.append(bx_node)
			find_flag = 1
			return new_node_set, batch_nodes
		# 对比监视节点变多，则选择成功，直接返回监视的节点结合和替换后的节点集合
		else:
			continue
	if find_flag == 0:
		batch_nodes.append(ex_node)
		# print '随机替换节点'
		# 尝试20次替换父节点无监视数增加，则进行所有批次topk节点(一共5*4-1=19个)的选择替换
		new_node_set, batch_nodes = choose_paralled_node(origin_node_set, node_preds, ex_node, pred_nodes, x_random, batch_nodes)
	return origin_node_set, batch_nodes

# 替换父节点失败，继续尝试替换所有批次的topk节点
def choose_paralled_node(origin_node_set, node_preds, ex_node, pred_nodes, x_random, batch_nodes):
	x_random = [node_degree[0] for node_degree in x_random]
	# 要选择除了ex_node之外的所有批次的topk节点
	if ex_node in x_random:
		x_random.remove(ex_node)
	origin_len = len(origin_node_set)
	for node in x_random:
		new_node_set = pred_nodes.union(set(node_preds[node]))
		# print origin_len, len(new_node_set)
		if len(new_node_set) > origin_len:
			batch_nodes.remove(ex_node)
			batch_nodes.append(node)
			return new_node_set, batch_nodes
	return origin_node_set, batch_nodes

# 初始化每个批次的节点，选取最前的topk*batch_num个节点进行随机分配批次
def get_random_batch(graph, topk, batch_num):
	out_degrees = []
	node_preds = {}
	for node in graph.nodes():
		out_degrees.append((node, graph.out_degree(node)))
		node_preds[node] = graph.predecessors(node)
	out_degrees_sorted = sorted(out_degrees, key=lambda x: x[1], reverse=True)
	topk_x_batch_nodes = out_degrees_sorted[:topk*batch_num]
	# print topk_x_batch_nodes
	# 由于刚开始度的分配是随机的，所以此处直接采用节点序号排序可以实现随机效果
	random_nodes = sorted(topk_x_batch_nodes, key=lambda x: x[0], reverse=False)
	# print random_nodes
	return node_preds, out_degrees, random_nodes

# 蜂群算法主函数，最大迭代次数为20
def Bee_Colony_process(graph, topk=5, batch_num=4, max_find_times = 20):
	node_preds, out_degrees, random_nodes = get_random_batch(graph, topk, batch_num)
	x_random = random_nodes
	node_batch = [random_nodes[i*topk:(i+1)*topk] for i in range(batch_num)]
	max_num = 0
	# while max_pro_num > 0:
	# 	max_pro_num -=1
	cnt = max_find_times
	pred_nodes,  all_batch, batch_max_num = [], [], []
	for i in range(batch_num):
		batch_nodes = []
		for tnode in node_batch[i]:
			node = tnode[0]
			batch_nodes.append(node)
			predecesors = node_preds[node]
			# print(predecesors)
			pred_nodes.extend(predecesors)
		pred_nodes_set = set(pred_nodes)
		batch_max_num.append(len(pred_nodes_set))
		all_batch.append(batch_nodes)
	max_nodes_set = set()
	monitor_max_nodes = []
	while cnt > 0:
		cnt = cnt - 1
		for i in range(batch_num):
			# print str(i) + '次'
			max_nodes_num, try_time = 0, topk
			# print pred_nodes_set
			while max_nodes_num < len(pred_nodes_set) and try_time > 0:
				choose_ex = random.randint(1, topk)
				ex_node = all_batch[i][choose_ex-1]
				# print ex_node
				# print batch_nodes
				node_set, new_batch_nodes =  choose_node(node_preds, ex_node, all_batch[i], x_random)
				max_nodes_num = len(node_set)
				if max_nodes_num > batch_max_num[i]:
					# print "再次最大:" + str(max_nodes_num)
					batch_max_num[i] = max_nodes_num
					all_batch[i] = new_batch_nodes
					if max_nodes_num > max_num:
						max_num = max_nodes_num
						print "当前最大得到提升:" + str(max_num)
						print new_batch_nodes
						monitor_max_nodes = all_batch[i]
						max_nodes_set = node_set
				try_time = try_time - 1
	print max_nodes_set
	return max_nodes_set, monitor_max_nodes


if __name__ == '__main__':
	graph = read_graph(graph_name="")
	# 进行对比，发现贪心算法和蜂群算法最大监视节点数会一致
	a = Out_Degree_Max.OutDegreeMax(graph)
	b = Greedy_Policy.GreedyPolicy(graph)
	max_nodes_set, monitor_max_nodes = Bee_Colony_process(graph)
	print '最大算法 ' + str(len(a))
	print '贪心算法 ' + str(len(b))
	print '蜂群算法 ' + str(len(max_nodes_set))
