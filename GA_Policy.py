#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
☆*°☆*°(∩^o^)~━━  2017/10/6 15:50        
      (ˉ▽￣～) ~~ 一捆好葱 (*˙︶˙*)☆*°
      Fuction：  遗传算法以批次为单位进行编码      √ ━━━━━☆*°☆*°
"""
import networkx as nx
import random
import Out_Degree_Max
import Greedy_Policy
import Bee_Colony_policy
graph_node_num, topk, batch_num= 250, 5, 10
actual_bits_num = len(bin(graph_node_num)[2:])
node_preds = {}
out_degrees = []
def read_graph(graph_name):
	graph = nx.DiGraph()
	ba_graph = nx.random_graphs.barabasi_albert_graph(graph_node_num, 3)
	graph = ba_graph.to_directed()
	return graph
# 检测批次是否有重复节点，仅用于测试
def identify_batch(batch_encode):
	bacth_sets = [batch_encode[i*actual_bits_num:(i+1)*actual_bits_num] for i in range(topk)]
	node_dec_values = [binary_decode(node) for node in bacth_sets]
	for value in node_dec_values:
		if value >= graph_node_num:
			return False
	node_set = set(node_dec_values)
	if len(node_set) < topk:
		return False
	else:
		return True
# 节点的二进制编码和解码
def binary_encode(node):
	global actual_bits_num
	bit_encode = bin(node)[2:]
	bits_dis = actual_bits_num - len(bit_encode)
	if bits_dis > 0:
		return bits_dis * '0' + bit_encode
	else:
		return bit_encode
def binary_decode(node_encode):
	first_flag, dec_value = False, 0
	for i in range(len(node_encode)):
		if node_encode[i] == '1':
			dec_value = dec_value * 2 +1
			if first_flag == False:
				first_flag =True
				dec_value = 1
		else:
			dec_value *= 2
	return dec_value
# 批次的二进制编码和解码
def batch_encode(batch):
	global actual_bits_num
	batch_dna = ''
	for node in batch:
		bit_encode = bin(node)[2:]
		bits_dis = actual_bits_num - len(bit_encode)
		if bits_dis > 0:
			batch_dna += bits_dis * '0' + bit_encode
		else:
			batch_dna += bit_encode
	return batch_dna
def batch_decode(batch_dna):
	batch_nodes_encode = [batch_dna[i*actual_bits_num:(i+1)*actual_bits_num] for i in range(topk)]
	batch_node_dec = []
	for node_encode in batch_nodes_encode:
		first_flag, dec_value = False, 0
		for i in range(len(node_encode)):
			if node_encode[i] == '1':
				dec_value = dec_value * 2 + 1
				if not first_flag:
					first_flag =True
					dec_value = 1
			else:
				dec_value *= 2
		batch_node_dec.append(dec_value)
	return batch_node_dec
# 包含除重的边界处理，无跨越边界的除重在交叉互换函数里面处理
def spe_pro(batch):
	for i in range(topk):
		node = batch[i]
		while node >= graph_node_num:
			t_ran = random.randint(node - graph_node_num + 1, node)
			# 确保topk个节点满足范围且不重复
			if not batch.count(batch[i] - t_ran) == 0:
				continue
			elif batch[i] - t_ran < graph_node_num:
				batch[i] -= t_ran
				node = batch[i]
	return batch

# 通过概率轮盘赌选择生存的群体，保证每一代数量一致
def select_survival(after_GA_batchs):
	sorted_batch = sorted(after_GA_batchs, key=lambda x: len(x[0]), reverse=True)[:batch_num]
	all_f_results = [len(f) for f, batch in sorted_batch]
	f_sum = sum(all_f_results)
	select_p = [float(f)/float(f_sum) for f in all_f_results]
	p_sums, sums =[0.0], 0.0
	all_batch = []
	for p in select_p:
		sums += p
		p_sums.append(sums)
	for i in range(batch_num):
		t = random.random()
		for j in range(len(p_sums)):
			if j>0 and t >= p_sums[j-1] and t < p_sums[j]:
				all_batch.append(sorted_batch[j-1][1])
				# print all_batch
				break
	max_monitor_set = sorted_batch[0][0]
	max_monitor_nodes = sorted_batch[0][1]
	# print [len(node_set) for node_set, select_batch in sorted_batch]
	return max_monitor_set, max_monitor_nodes, all_batch

# 对交叉互换后的各个批次的每个编码以一定概率var_p进行变异操作
# 该函数保证变异后的批次节点跟原批次节点有所不同
def batch_Mutation(node_batches, mut_p=0.1, max_try_time=topk * actual_bits_num):
	# for batch in node_batches:
	# 	if node_batches.count(batch) > int(round(batch_num * 0.9)):
	# 		mut_p = 0.3
	# 		node_batches.remove(batch)
	# 		node_batches.append(batch)
	# 		break
	for i in range(batch_num):
		mut_flag = False
		batch_en = batch_encode(node_batches[i])
		while not mut_flag and max_try_time > 0:
			new_dna, cnt, one_flag, bit = '', 0, False, ''
			max_try_time -= 1
			while cnt < actual_bits_num * topk:
				bit = ''
				if_var = random.random()
				# 对该位进行变异
				if if_var < mut_p:
					if batch_en[cnt] == '1':
						bit = '0'
					else:
						bit = '1'
				if (batch_en[cnt] == '1' and bit == '') or bit == '1':
					new_dna += '1'
				elif (batch_en[cnt] == '0' and bit == '') or bit == '0':
					new_dna += '0'
				cnt += 1
			# print batch_decode(new_dna)
			if identify_batch(new_dna):
				node_batches[i] = batch_decode(new_dna)
				mut_flag = True
	return node_batches



# f为目标函数,使f(batch_nodes)=len(nodes_set)最大化
def get_pred_node_set(batch_nodes):
	nodes_set = []
	for node in batch_nodes:
		nodes_set.extend(node_preds[node])
	nodes_set = set(nodes_set)
	return len(nodes_set), nodes_set


# 选择这一代的一定比例的群体进行染色体某一个位置后序列的交叉互换，含边界检测和除重操作，最大尝试次数为节点编码最长位数
def batch_ex_dna(node_batches, cro_p=0.25, max_try_time=actual_bits_num):
	select_to_ex = [False] * batch_num
	select_dnas = []
	ex_num, actual_select_num = int(round(batch_num * cro_p)), 0
	if not ex_num % 2 == 0:
		ex_num += 1
	# print 'rounnd:' + str(ex_num)
	while actual_select_num < ex_num:
		choose = random.randint(0, batch_num - 1)
		if not select_to_ex[choose]:
			select_to_ex[choose] = True
			select_dnas.append(batch_encode(node_batches[choose]))
			actual_select_num += 1
	new_batches, succ_flag = [], False
	for i in range(int(ex_num / 2)):
		while not succ_flag and max_try_time > 0:
			max_try_time -= 1
			t_pos = random.randint(1, topk * actual_bits_num - 2)
			new_dna_1 = select_dnas[i][:t_pos] + select_dnas[ex_num - i - 1][t_pos:]
			new_dna_2 = select_dnas[ex_num - i - 1][:t_pos] + select_dnas[i][t_pos:]
			new_batch_1, new_batch_2 = batch_decode(new_dna_1), batch_decode(new_dna_2)
			new_batch_1, new_batch_2 = spe_pro(new_batch_1), spe_pro(new_batch_2)
			if len(set(new_batch_1)) == topk and len(set(new_batch_2)) == topk:
				succ_flag = True
				try:
					node_batches.remove(batch_decode(select_dnas[i]))
				except:
					print node_batches
					print batch_decode(select_dnas[i])
				try:
					node_batches.remove(batch_decode(select_dnas[ex_num - 1 - i]))
				except:
					print node_batches
					print batch_decode(select_dnas[ex_num - 1 - i])
				node_batches.append(new_batch_1)
				node_batches.append(new_batch_2)
				# print '成功交叉互换——原：'
				# print batch_decode(select_dnas[i]), batch_decode(select_dnas[ex_num - 1 - i])
				# print '后：'
				# print new_batch_1, new_batch_2
	return node_batches

# 初始化每个批次的节点，选取最前的topk*batch_num个节点进行随机分配批次
def get_random_batch(graph):
	global node_preds, out_degrees
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

# 遗传算法主函数，最大迭代次数与变异率成反比，图越大越要提高
def GA(graph, max_iter_tiems = 3000):
	node_preds, out_degrees, random_nodes = get_random_batch(graph)
	node_batch = [random_nodes[i*topk:(i+1)*topk] for i in range(batch_num)]
	balance_num = 0
	cnt = max_iter_tiems
	pred_nodes_set, pred_nodes,  all_batch, batch_max_num = set(), [], [], []
	# pred_nodes_set用于暂时存放该批次节点最大监视集合，并将个数储存在batch_max_num中
	# all_batch存放各个批次的节点集合
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
	while cnt > 0 and balance_num < 10000:
		cnt = cnt - 1
		# print cnt
		after_GA_batchs = []
		# 先交叉互换
		for i in range(batch_num):
			old_set_size, old_node_set = get_pred_node_set(all_batch[i])
			after_GA_batchs.append((old_node_set, all_batch[i]))
		monitor_set, monitor_nodes, all_batch = select_survival(after_GA_batchs)
		all_batch = batch_ex_dna(all_batch)
		# 再选择群体变异
		all_batch = batch_Mutation(all_batch)
		for i in range(batch_num):
			new_set_size, new_node_set = get_pred_node_set(all_batch[i])
			after_GA_batchs.append((new_node_set, all_batch[i]))
		# 轮盘赌选择存留的群体
		monitor_set, monitor_nodes, all_batch = select_survival(after_GA_batchs)
		# print 'GA_batch:'
		# print after_GA_batchs
		# print all_batch
		if len(max_nodes_set) > len(monitor_set):
			balance_num += 1
		elif len(monitor_set) > len(max_nodes_set):
			print len(monitor_set)
			balance_num = 0
			monitor_max_nodes = monitor_nodes
			max_nodes_set = monitor_set
	# return max_nodes_set, monitor_max_nodes
	# print 'balan：' + str(balance_num)
	return max_nodes_set

if __name__ == '__main__':
	graph_node_num = 250 * (10**2)
	graph = read_graph(graph_name="")
	# a = Out_Degree_Max.OutDegreeMax(graph)
	b = Greedy_Policy.GreedyPolicy(graph)
	print '贪心算法 ' + str(len(b))
	# max_nodes_set, monitor_max_nodes = Bee_Colony_policy.Bee_Colony_process(graph)
	# print '蜂群算法 ' + str(len(max_nodes_set))
	print 'strat'
	d = GA(graph)
	print '遗传算法 ' + str(len(d))
	# print '最大算法 ' + str(len(a))
	# print '遗传算法 ' + str(len(d))
