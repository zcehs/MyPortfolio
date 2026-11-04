
import numpy as np
import heapq
import copy
from typing import List
from event import Event
from Resource import Resource_Event
from DT import DT
from DNN import DNN
from config import (
    BS_STORAGE,
    BS_COMPUTING,
    BS_BANDWIDTH,
	INF,
	TIME_SLOT,
    #DT model
    NUM_DT,
	MAX_DT_SIZE,
	IOT_STATIC,
	IOT_MOVE_PROB,
    #DNN model
	DNN_SIZE,
	MAX_ACC,
	MIN_ACC,
	BETA_ACC,
    # Event model
    EVENT_DENSITY,
	MAX_DELAY_CONSTRAINT,
    # Netwrok model
    MAX_NODES,
    MEC_NODE_GRAPH, MEC_LINK_GRAPH, LINK_PROPAGATION , MAX_LINK
)
import pprint

class ENV(object):
	def __init__(self, users: int):

		self.users = users

		self.user_id = 0
		
		#self.dt_s_dim = MAX_NODES + 2 * MAX_LINK + 3 + MAX_NODES
		self.dt_s_dim = MAX_NODES + 2 * MAX_LINK + 4
		self.dnn_s_dim = MAX_NODES + 2 * MAX_LINK + 3 + NUM_DT * 2

		self.observation_space = None
		self.cur_state_ = None
		self.next_state = None

		self.a_dim = MAX_NODES	                        # Action output number
		self.a_bound = 1 								# f = (0,1]	output range


		
		#----------------
		#event setting
		#----------------
		self.event_queue: List[Event] = []
		self.resource_event_queue: List[Resource_Event] = []
		self.event: Event = None

		self.updating = False
		self.updating_DT = 0
		#----------------
		#time
		#----------------
		self.time = 0
		self.time_slot = 0
		self.arrival_time = []
		self.finish_time = []
		#----------------
		#DT setting
		#----------------
		self.DT_lst: List[DT] = []
		self.average_AoI = []
		self.req_count = []
		self.is_data_arrival = False
		self.mig_num = [0 for i in range(NUM_DT)]
		#----------------
		#DNN setting
		#----------------
		self.pre_user = -1
		self.DNN_lst: List[DNN] = []
		self.service_downtime = 0
		self.estimated_result_delay = 0
		self.estimated_acc = 0
		self.acc = 0
		self.denined_DNN = 0
		self.finished_DNN_count = 0
		#----------------
		#env state
		#----------------
		self.Node_storage: List[int] = []							# All of the nodes storage resource
    


	# start->step->step...->reset(start to next episode)->step->step->...
	def reset(self):
		
		#----------------
		#event setting
		#----------------
		self.user_id = 0
		self.finish_time = []
		self.mig_num = [0 for i in range(NUM_DT)]

		self.time = 0
		self.time_slot = 0
		self.event_queue: List[Event] = []
		self.resource_event_queue: List[Resource_Event] = []
		self.event: Event = None

		self.arrival_time = []
		

		self.updating = True
		self.updating_DT = 0


		#----------------
		#env state
		#----------------
		self.Node_storage = [BS_STORAGE for i in range(MAX_NODES)]	

		self.node_state = [0 for i in range(MAX_NODES)]
		self.link_state = [[0 for j in range(2)]for i in range(MAX_LINK)]
		#----------------
		#DT setting
		#----------------
		self.DT_lst = [DT.new_DT(i) for i in range(NUM_DT)]
		self.average_AoI = [0 for i in range(NUM_DT)]
		self.req_count = [0 for i in range(NUM_DT)]
		self.is_data_arrival = False

		for dt in self.DT_lst:
			self.adjust_node_storage(dt.DT_size, [dt.node], 0)
			self.adjust_node_state(dt.proc_req, [dt.node], 0)
			dt.update_link, dt.update_link_dir = self.find_shortest_path(
				dt.dr_req, 
				dt.IoT_node, 
				dt.node, 
				dt.update_size)
					
			self.adjust_link_state(dt.dr_req, dt.update_link, dt.update_link_dir, 0)
			#dt.AoI = self.count_AoI(dt.DT_id)
			"""dt.update_delay = self.count_update_delay(dt.DT_id)
			dt.data_arrival_time.append(dt.update_delay + self.time_slot * TIME_SLOT)"""
		#----------------
		#DNN setting
		#----------------
		self.pre_user = -1
		self.DNN_lst: List[DNN] = []
		self.service_downtime = 0
		self.estimated_result_delay = 0
		self.estimated_acc = 0
		self.acc = 0
		self.denined_DNN = 0
		self.finished_DNN_count = 0

		"""if self.updating:
			self.get_state(0)
		else:
			self.get_state(1)

		self.cur_state_ = None
		self.next_state = copy.deepcopy(self.observation_space)
		

		return self.next_state"""

	def check_data_arrival(self):
		if not self.is_data_arrival:
			for i in range(NUM_DT):
				for j in range(self.time_slot + 1):
					if self.time < self.DT_lst[i].data_arrival_time[self.time_slot - j]:
						self.is_data_arrival = False
					else:
						self.is_data_arrival = True
						break
				if not self.is_data_arrival:
					break


	def next_time_slot(self):
		self.time_slot += 1
		"""for i in range(NUM_DT):
			self.DT_lst[i].update_delay = self.count_update_delay(self.DT_lst[i].DT_id)
			self.DT_lst[i].data_arrival_time.append(self.DT_lst[i].update_delay + self.time_slot * TIME_SLOT)
			#self.DT_lst[i].reset_AoI()"""
		#self.time = self.time_slot * TIME_SLOT
		self.updating = True
		self.finish_event(self.time_slot * TIME_SLOT)
		self.Iot_move()
		return self.time_slot * TIME_SLOT
	
	def next_event(self):
		time = 0
		if (len(self.event_queue) == 0):
			if not self.is_data_arrival:
				time = self.next_time_slot()
				return
			self.generate_arrivals(self.users, self.time)
			for i in range(self.users):
				self.generate_new_event(self.user_id)
				self.user_id = self.user_id + 1	
			
		if self.updating:
			if self.event_queue[0].time < self.time_slot * TIME_SLOT + TIME_SLOT:
				#self.time = self.event_queue[0].time
				time = self.event_queue[0].time
				self.event = self.event_queue[0]
				self.event_queue.remove(self.event)
				self.updating = False
				self.finish_event(time)
				return
		else:
			if self.time // TIME_SLOT == self.event_queue[0].time // TIME_SLOT:
				#self.time = self.event_queue[0].time
				time = self.event_queue[0].time
				self.event = self.event_queue[0]
				self.event_queue.remove(self.event)
				self.finish_event(time)
				return
		time = self.next_time_slot()

		
		
			
		
	
	def generate_arrivals(self, event_num, base_time):
		inter_arrival = np.random.exponential(EVENT_DENSITY / event_num, event_num)
		arrival = (base_time + np.cumsum(inter_arrival)).tolist()
		self.arrival_time = arrival
	
	def generate_new_event(self, user):
		if len(self.arrival_time) == 0:
			self.generate_arrivals(self.users, ((self.time - 1) // EVENT_DENSITY + 1) * EVENT_DENSITY)
		while(self.arrival_time[0] < self.time):
			self.arrival_time.pop(0)
			self.denined_DNN += 1
			if len(self.arrival_time) == 0:
				self.generate_arrivals(self.users, ((self.time - 1) // EVENT_DENSITY + 1) * EVENT_DENSITY)
		event = Event.new_Event(user=user, time=self.arrival_time[0])
		self.event_queue.append(event)
		self.arrival_time.pop(0)


	# Find the maximum priority node in action
	def find_max_prior_node(self, action, deploy_size):
		
		c_node = -1
		max_prior = -1

		for j in range(self.a_dim):#self.a_dim is MAX_NODES
			
			if(self.Node_storage[j] < deploy_size):#index is which VNF #whether node storage is less than storage requirement or not
				continue


			if(action[0][j] > max_prior):
				c_node = j
				max_prior = action[0][j]

			elif(action[0][j] == max_prior):
				c_node = self.tie_breaker(c_node, j)
				max_prior = action[0][c_node]

		if c_node == -1:
			print("No node can deploy!")
			if deploy_size == DNN_SIZE:
				print("DNN step")
			else:
				print("DT step")
			print(self.Node_storage)
			exit()
		return c_node
	
	def tie_breaker(self, node1, node2):
		if(self.Node_storage[node1] > self.Node_storage[node2]):
			return node1
		elif(self.Node_storage[node1] < self.Node_storage[node2]):
			return node2
		else:
			return np.random.choice([node1, node2])
	
	# Dijstra algorithmn
	def find_shortest_path(self, dr_req, src, dst, packetSize):
		"""
		return:
			dst_path: 經過的 link id list
			dst_dir : 每條 link 的方向 list
					0 表示 u < v
					1 表示 u > v
		"""

		if src == dst:
			return [], []

		dist = np.full(MAX_NODES, INF, dtype=float)
		visited = np.zeros(MAX_NODES, dtype=bool)

		# prev_node[v] = 到 v 的前一個 node
		prev_node = [-1] * MAX_NODES

		# prev_link[v] = 前一個 node 到 v 使用的 link id
		prev_link = [-1] * MAX_NODES

		# prev_dir[v] = 前一個 node 到 v 的方向
		prev_dir = [-1] * MAX_NODES

		dist[src] = 0

		# heap 內容: (目前距離, node)
		heap = [(0, src)]

		while heap:
			cur_dist, u = heapq.heappop(heap)

			if visited[u]:
				continue

			visited[u] = True

			if u == dst:
				break

			for idx, v in enumerate(MEC_NODE_GRAPH[u]):
				link_id = MEC_LINK_GRAPH[u][idx]

				propagation_delay = LINK_PROPAGATION[link_id]

				if u < v:
					direction = 0
				else:
					direction = 1

				# 避免除以 0
				if dr_req <= 0:
					continue

				data_rate = BS_BANDWIDTH * dr_req / (self.link_state[link_id][direction] + dr_req)

				if data_rate <= 0:
					continue

				transmission_delay = packetSize / data_rate * 1000

				edge_cost = propagation_delay + transmission_delay

				new_dist = cur_dist + edge_cost

				if new_dist < dist[v]:
					dist[v] = new_dist
					prev_node[v] = u
					prev_link[v] = link_id
					prev_dir[v] = direction
					heapq.heappush(heap, (new_dist, v))

		# 找不到路徑
		if dist[dst] == INF:
			return [], []

		# 從 dst 回推到 src
		dst_path = []
		dst_dir = []

		cur = dst

		while cur != src:
			dst_path.append(prev_link[cur])
			dst_dir.append(prev_dir[cur])
			cur = prev_node[cur]

			if cur == -1:
				return [], []

		# 因為是從 dst 回推，所以要反轉
		dst_path.reverse()
		dst_dir.reverse()

		return dst_path, dst_dir
	
	
	def count_proc_delay(self, proc_req, porc_size, node):
		proc_delay = 0
		if(self.node_state[node] == 0):
			print("node state error 526")
			exit()
		proc_rate = BS_COMPUTING * proc_req / self.node_state[node]

		proc_delay = porc_size / proc_rate * 1000 # Mbit / Gbps (ms)

		return proc_delay
	
	def count_prop_delay(self, dr_req, packetSize, candid_link, candid_link_dir):
		prop_delay = 0
		trans_delay = 0
		paths, path_dirs = self.normalize_link_paths(candid_link, candid_link_dir)
		for path, dirs in zip(paths, path_dirs):
			if len(path) == 0:
				continue

			if len(path) != len(dirs):
				raise ValueError(f"Link and direction length mismatch: "f"{len(path)} != {len(dirs)}")

			min_data_rate = float("inf")

			for link_id, direction in zip(path, dirs):
				used_bandwidth = self.link_state[link_id][direction]

				if used_bandwidth <= 0:
					raise RuntimeError(f"Invalid link state: link={link_id}, "f"direction={direction}, "f"state={used_bandwidth}")

				data_rate = (BS_BANDWIDTH * dr_req / used_bandwidth)

				min_data_rate = min(min_data_rate, data_rate)
				prop_delay += LINK_PROPAGATION[link_id]

			if min_data_rate != float("inf"):
				trans_delay += packetSize / min_data_rate * 1000
	
		return prop_delay + trans_delay
	
	def count_AoI(self, DT_id):
		time_slot = self.time // TIME_SLOT
		if self.time < (self.DT_lst[DT_id].update_delay + TIME_SLOT * time_slot):
			time_slot -= 1
		AoI = self.time - TIME_SLOT * time_slot
		return AoI

	def count_update_delay(self, DT_id, mig_time):
		DT = self.DT_lst[DT_id]
		proc_delay = self.count_proc_delay(DT.proc_req, DT.update_size, DT.node)
		prop_delay = self.count_prop_delay(DT.dr_req, DT.update_size, DT.update_link, DT.update_link_dir)
		update_delay = max(prop_delay, mig_time) + proc_delay
		return update_delay

	def count_acc(self, dnn):
		acc = (MIN_ACC + (MAX_ACC - MIN_ACC) * np.exp(-BETA_ACC * dnn.AoI / TIME_SLOT))
		return acc

	def count_train_prop_delay(self, dnn, mode = 1):
			train_prop_delay_lst = [0 for i in range(NUM_DT)]
			dt_id = 0
			AoI_lst = []
			for dt_is_used in dnn.DT_req_lst:
				if dt_is_used:
					train_prop_delay = self.count_prop_delay(dnn.dr_req, self.DT_lst[dt_id].train_data_size, dnn.train_data_link[dt_id], dnn.train_data_link_dir[dt_id])
					train_prop_delay_lst[dt_id] = train_prop_delay
					if mode == 1:
						AoI = -1
						for i in range(self.time_slot + 1):
							if self.time >= self.DT_lst[dt_id].data_arrival_time[self.time_slot - i]:
								AoI = (self.time - (self.time_slot - i) * TIME_SLOT) + train_prop_delay
								"""print("DT", dt_id, "data arrival time:", (self.time_slot - i) * TIME_SLOT)
								print("time:", self.time)
								print("time_slot:", self.time_slot)
								print("train_prop_delay:", train_prop_delay)
								print("AoI:", AoI)
								print("update_delay:", self.DT_lst[dt_id].update_delay)"""
								break
						if AoI == -1:
							print("DT", dt_id, "data arrival time:", self.DT_lst[dt_id].data_arrival_time)
							print("time:", self.time)
							print("time_slot:", self.time_slot)
							exit()
						AoI_lst.append(AoI)
						self.DT_lst[dt_id].count_AoI(AoI)
						self.average_AoI[dt_id] += AoI
						self.req_count[dt_id] += 1
				dt_id += 1
			train_prop_delay = max(train_prop_delay_lst)
			dnn.train_prop_delay_lst = train_prop_delay_lst
			if mode == 1:
				dnn.AoI = max(AoI_lst)
			return train_prop_delay

	def count_result_delay(self, dnn):
		train_prop_delay_lst = []
		train_proc_delay = 0
		result_prop_delay = 0
		dt_id = 0
		for dt_is_used in dnn.DT_req_lst:
			if dt_is_used:
				train_prop_delay = self.count_prop_delay(dnn.dr_req, self.DT_lst[dt_id].train_data_size, dnn.train_data_link[dt_id], dnn.train_data_link_dir[dt_id])
				train_prop_delay_lst.append(train_prop_delay)
			dt_id += 1
		train_prop_delay = max(train_prop_delay_lst)
		train_proc_delay = self.count_proc_delay(dnn.proc_req, dnn.train_data_size, dnn.node)
		result_prop_delay = self.count_prop_delay(dnn.dr_req, dnn.result_size, dnn.result_link, dnn.result_link_dir)
		#print("user", user, "train_prop_delay:", train_prop_delay, "train_proc_delay:", train_proc_delay, "result_prop_delay:", result_prop_delay, "result_delay:", result_delay)
		return train_prop_delay, train_proc_delay, result_prop_delay

	def count_result_delay_with_AoI(self, dnn):
		dt_id = 0
		AoI_lst = []
		for dt_is_used in dnn.DT_req_lst:
			if dt_is_used:
				train_prop_delay = self.count_prop_delay(dnn.dr_req, self.DT_lst[dt_id].train_data_size, dnn.train_data_link[dt_id], dnn.train_data_link_dir[dt_id])
				AoI = -1
				for i in range(self.time_slot + 1):
					if self.time >= self.DT_lst[dt_id].data_arrival_time[self.time_slot - i]:
						AoI = (self.time - (self.time_slot - i) * TIME_SLOT) + train_prop_delay
						"""print("DT", dt_id, "data arrival time:", (self.time_slot - i) * TIME_SLOT)
						print("time:", self.time)
						print("time_slot:", self.time_slot)
						print("train_prop_delay:", train_prop_delay)
						print("AoI:", AoI)
						print("update_delay:", self.DT_lst[dt_id].update_delay)"""
						break
				if AoI == -1:
					print("DT", dt_id, "data arrival time:", self.DT_lst[dt_id].data_arrival_time)
					print("time:", self.time)
					print("time_slot:", self.time_slot)
					exit()
				AoI_lst.append(AoI)
				self.DT_lst[dt_id].count_AoI(AoI)
				self.average_AoI[dt_id] += AoI
				self.req_count[dt_id] += 1
			dt_id += 1
		dnn.AoI = max(AoI_lst)
	
	def count_new_end_time(self, old_time, old_end_time, new_end_time, remaining_work):
		complete_rate = 0
		if (not old_end_time == None) and (not old_end_time == old_time):
			complete_rate = (self.time - old_time) / (old_end_time - old_time)
		complete_rate = np.clip(complete_rate, 0.0, 1.0)
		remaining_work *= (1 - complete_rate)
		new_end_time = new_end_time * remaining_work
		
		return max(new_end_time, 0), remaining_work

	def estimate_result_delay(self, dnn):
		train_prop_delay = dnn.train_prop_delay
		if dnn.phase == 0:
			train_proc_delay = self.estimate_proc_delay(dnn.proc_req, dnn.train_data_size, dnn.node)
		else:
			train_proc_delay = self.count_proc_delay(dnn.proc_req, dnn.train_data_size, dnn.node)
		result_link = []
		result_link_dir = []
		result_link, result_link_dir = self.find_shortest_path(dnn.dr_req,
														dnn.node, 
														dnn.dst_node, 
														dnn.result_size)
		result_prop_delay = self.estimate_prop_delay(dnn.dr_req, dnn.result_size, result_link, result_link_dir)
		#print("user", user, "train_prop_delay:", train_prop_delay, "train_proc_delay:", train_proc_delay, "result_prop_delay:", result_prop_delay, "result_delay:", result_delay)
		return train_prop_delay, train_proc_delay, result_prop_delay

	def estimate_proc_delay(self, proc_req, porc_size, node):
		proc_delay = 0

		proc_rate = BS_COMPUTING * proc_req / (self.node_state[node] + proc_req)

		proc_delay = porc_size / proc_rate * 1000 # Mbit / Gbps (ms)

		return proc_delay
		
	def estimate_prop_delay(self, dr_req, packetSize, candid_link, candid_link_dir):
		prop_delay = 0
		trans_delay = 0
		paths, path_dirs = self.normalize_link_paths(candid_link, candid_link_dir)
		for path, dirs in zip(paths, path_dirs):
			if len(path) == 0:
				continue

			if len(path) != len(dirs):
				raise ValueError(f"Link and direction length mismatch: "f"{len(path)} != {len(dirs)}")

			min_data_rate = float("inf")

			for link_id, direction in zip(path, dirs):
				used_bandwidth = self.link_state[link_id][direction]

				data_rate = (BS_BANDWIDTH * dr_req / (used_bandwidth + dr_req))

				min_data_rate = min(min_data_rate, data_rate)
				prop_delay += LINK_PROPAGATION[link_id]

			if min_data_rate != float("inf"):
				trans_delay += packetSize / min_data_rate * 1000
	
		return prop_delay + trans_delay

	def add_finish_time(self, type, id, count, finish_time):
		# 移除同一個 job 舊的 completion time
		for i in range(len(finish_time)):
			self.finish_time = [
				ft for ft in self.finish_time
				if not (ft[0] == type and ft[1] == id and ft[2] == count + i)
			]
			if finish_time[i] <= self.time:
				continue

			self.finish_time.append((type, id, count + i, finish_time[i]))
		self.finish_time.sort(key=lambda x: x[3])


	def deploy_DNN(self, dnn):
		phase_time = 0
		count = 0
		end_time = []
		if dnn.phase == 0:
			train_data_link = [[] for i in range(NUM_DT)]
			train_data_link_dir = [[] for i in range(NUM_DT)]
			train_data_size = 0

			dt_id = 0
			train_prop_delay = 0
			for dt_is_used in dnn.DT_req_lst:
				if dt_is_used:
					c_link = []
					c_link_dir = []
					if self.DT_lst[dt_id].node == dnn.node:
						dnn.clear_train_data_path(dt_id)
					else:
						c_link, c_link_dir = self.find_shortest_path(dnn.dr_req,
																	self.DT_lst[dt_id].node, 
																	dnn.node, 
																	self.DT_lst[dt_id].train_data_size)
					train_data_link[dt_id] = (c_link)
					train_data_link_dir[dt_id] = (c_link_dir)
					train_data_size = train_data_size + self.DT_lst[dt_id].train_data_size
				dt_id += 1

			dnn.set_train_data_size(train_data_size)
			dnn.train_data_link = train_data_link
			dnn.train_data_link_dir = train_data_link_dir	
			self.adjust_link_state(dnn.dr_req, dnn.train_data_link, dnn.train_data_link_dir, 0)#allocate DT to DNN path
			dnn.train_prop_delay = self.count_train_prop_delay(dnn)
			dnn.acc = self.count_acc(dnn)
			phase_time = dnn.train_prop_delay
			for i in range(NUM_DT):
				end_time.append(self.time + dnn.train_prop_delay_lst[i])
			if not any(dnn.pending_DT_lst):
				dnn.phase = 1
				dnn.remaining_work[0] = 1.0

				self.deploy_DNN(dnn)
				return
		elif dnn.phase == 1:
			self.adjust_node_state(dnn.proc_req, [dnn.node], 0)
			dnn.train_proc_delay = self.count_proc_delay(dnn.proc_req, dnn.train_data_size, dnn.node)
			phase_time = dnn.train_proc_delay
			end_time.append(self.time + phase_time)
			count = NUM_DT - 1 + dnn.phase
		elif dnn.phase == 2:
			result_link = []
			result_link_dir = []
			result_link, result_link_dir = self.find_shortest_path(dnn.dr_req,
															dnn.node, 
															dnn.dst_node, 
															dnn.result_size)
			
			dnn.result_link = result_link
			dnn.result_link_dir = result_link_dir

			self.adjust_link_state(dnn.dr_req, dnn.result_link, dnn.result_link_dir, 0)#allocate DNN to user path
			dnn.result_prop_delay = self.count_prop_delay(dnn.dr_req, dnn.result_size, dnn.result_link, dnn.result_link_dir)
			phase_time = dnn.result_prop_delay
			end_time.append(self.time + phase_time)
			count = NUM_DT - 1 + dnn.phase


		dnn.res_time = self.time + phase_time

		dnn.time = self.time

		self.add_finish_time(1, dnn.user, count, end_time)
	
	def update_DT(self, DT_id, candid_node):
			
		self.adjust_link_state(self.DT_lst[DT_id].dr_req, self.DT_lst[DT_id].update_link, self.DT_lst[DT_id].update_link_dir, 1)

		self.DT_lst[DT_id].node = candid_node

		self.DT_lst[DT_id].update_link, self.DT_lst[DT_id].update_link_dir = self.find_shortest_path(
				self.DT_lst[DT_id].dr_req, 
				self.DT_lst[DT_id].IoT_node, 
				self.DT_lst[DT_id].node, 
				self.DT_lst[DT_id].update_size)
		
		self.adjust_link_state(self.DT_lst[DT_id].dr_req, self.DT_lst[DT_id].update_link, self.DT_lst[DT_id].update_link_dir, 0)


		
	
	def update_end_time(self):
		for resource_event in self.resource_event_queue:
			if resource_event.res_time <= self.time:
				continue
			new_mig_time = self.count_prop_delay(resource_event.dr_req, resource_event.size, resource_event.link, resource_event.link_dir)
			#print("DT_mig", "resource_event.time:", resource_event.time, "resource_event.res_time:", resource_event.res_time, "new_mig_time:", new_mig_time)
			new_mig_time, resource_event.remaining_work = self.count_new_end_time(resource_event.time, resource_event.res_time, new_mig_time, resource_event.remaining_work)
			resource_event.res_time = self.time + new_mig_time
			if resource_event.mig_num == self.mig_num[resource_event.user] - 1:
				self.DT_lst[resource_event.user].mig_end_time = self.time + new_mig_time
			self.add_finish_time(0, resource_event.user, resource_event.mig_num, [resource_event.res_time])
			resource_event.time = self.time

		for dt_id in range(NUM_DT):
			remaining_mig_time = max(self.DT_lst[dt_id].mig_end_time - self.time, 0)
			self.DT_lst[dt_id].update_delay = self.count_update_delay(dt_id, remaining_mig_time)

		for dnn in self.DNN_lst:
			if dnn.res_time <= self.time:
				continue
			new_phase_time = 0
			end_time = []
			count = 0
			if dnn.phase == 0:
				old_delay_lst = dnn.train_prop_delay_lst.copy()
				self.count_train_prop_delay(dnn, 0)
				for i in range(NUM_DT):
					if dnn.pending_DT_lst[i]:
						dnn.train_prop_delay_lst[i], dnn.remaining_work[i] = self.count_new_end_time(dnn.time, dnn.time + old_delay_lst[i], dnn.train_prop_delay_lst[i], dnn.remaining_work[i])
				new_phase_time = max(dnn.train_prop_delay_lst)
				for i in range(NUM_DT):
					end_time.append(self.time + dnn.train_prop_delay_lst[i])
			elif dnn.phase == 1:
				new_phase_time = self.count_proc_delay(dnn.proc_req, dnn.train_data_size, dnn.node)
				new_phase_time, dnn.remaining_work[0] = self.count_new_end_time(dnn.time, dnn.res_time, new_phase_time, dnn.remaining_work[0])
				end_time.append(self.time + new_phase_time)
				count = NUM_DT - 1 + dnn.phase
			elif dnn.phase == 2:
				new_phase_time = self.count_prop_delay(dnn.dr_req, dnn.result_size, dnn.result_link, dnn.result_link_dir)
				new_phase_time, dnn.remaining_work[0] = self.count_new_end_time(dnn.time, dnn.res_time, new_phase_time, dnn.remaining_work[0])
				end_time.append(self.time + new_phase_time)
				count = NUM_DT - 1 + dnn.phase
			dnn.result_delay += self.time - dnn.time
			dnn.res_time = self.time + new_phase_time
			self.add_finish_time(1, dnn.user, count, end_time)
			dnn.time = self.time
	
	def adjust_node_storage(self, size, node, mode):
		if mode == 0:
			if (node):
				for n in node:
					self.Node_storage[n] = self.Node_storage[n] - size
					if(self.Node_storage[n] < 0):
						print("Node storage error in allocate!")
						exit(1)
		elif mode == 1:
			if (node):
				for n in node:
					self.Node_storage[n] = self.Node_storage[n] + size
					if(self.Node_storage[n] > BS_STORAGE):
						print("Node storage error in release!")
						exit(1)
		else:
			print("mode error!")
			exit(1)

	def adjust_node_state(self, proc_req, node, mode):
		if mode == 0:
			if (node):
				for n in node:
					self.node_state[n] = self.node_state[n] + proc_req
		elif mode == 1:
			if (node):
				for n in node:
					self.node_state[n] = self.node_state[n] - proc_req
		else:
			print("mode error!")
			exit(1)

	def normalize_link_paths(self, links, dirs):
		"""
		將單一路徑轉換成多路徑格式。

		單一路徑：
			links = [1, 2, 3]
			dirs  = [0, 1, 0]

		多條路徑：
			links = [[1, 2], [3, 4]]
			dirs  = [[0, 1], [1, 0]]
		"""

		if links is None or dirs is None:
			return [], []

		if len(links) == 0:
			return [], []

		# links[0] 是整數，代表目前是單一路徑
		if isinstance(links[0], (int, np.integer)):
			if len(links) != len(dirs):
				raise ValueError(
					f"Link and direction length mismatch: "
					f"{len(links)} != {len(dirs)}"
				)

			return [links], [dirs]

		# links[0] 是 list，代表已經是多路徑
		if len(links) != len(dirs):
			raise ValueError(
				f"Path count mismatch: {len(links)} != {len(dirs)}"
			)

		return links, dirs

	def adjust_link_state(self, dr_req, links, dirs, mode):
		links, dirs = self.normalize_link_paths(links, dirs)
		if mode == 0:
			change = dr_req       # 分配頻寬
		elif mode == 1:
			change = -dr_req      # 釋放頻寬
		else:
			raise ValueError(f"Invalid link adjustment mode: {mode}")

		for path, path_dirs in zip(links, dirs):

			if len(path) != len(path_dirs):
				raise ValueError(
					f"Link and direction length mismatch: "
					f"{len(path)} != {len(path_dirs)}"
				)

			for link_id, direction in zip(path, path_dirs):
				self.link_state[link_id][direction] += change

				if self.link_state[link_id][direction] < 0:
					raise RuntimeError(
						f"Link state became negative: "
						f"link={link_id}, direction={direction}, "
						f"value={self.link_state[link_id][direction]}"
					)



		
	def finish_event(self, time):

		while self.finish_time and self.finish_time[0][3] <= time:

			current_time = self.finish_time[0][3]
			self.time = current_time
			current_events = []
			while (self.finish_time and abs(self.finish_time[0][3] - current_time) < 1e-9):
				current_events.append(self.finish_time.pop(0))
			for ft in current_events:
				if ft[0] == 0:
					unfinished_resource_events = []
					for resource_event in self.resource_event_queue:
						if (resource_event.user == ft[1] and resource_event.mig_num == ft[2]):
							# migration 完成，釋放 bandwidth
							self.adjust_link_state(resource_event.dr_req, resource_event.link, resource_event.link_dir, 1)
						else:
							unfinished_resource_events.append(resource_event)
					self.resource_event_queue = unfinished_resource_events
				else:
					unfinished_DNN = []
					for dnn in self.DNN_lst:
						if dnn.user != ft[1]:
							unfinished_DNN.append(dnn)
							continue
						if dnn.phase == 0:
							dt_id = ft[2]
							self.adjust_link_state(dnn.dr_req, dnn.train_data_link[dt_id], dnn.train_data_link_dir[dt_id], 1)
							dnn.clear_train_data_path(dt_id)
							if any(dnn.pending_DT_lst):
								dnn.phase = 0
							else:
								dnn.result_delay += (self.time - dnn.time)
								dnn.phase = 1
								dnn.remaining_work[0] = 1.0
								self.deploy_DNN(dnn)
							unfinished_DNN.append(dnn)
						elif dnn.phase == 1:
							self.adjust_node_state(dnn.proc_req, [dnn.node], 1)

							dnn.result_delay += (self.time - dnn.time)
							dnn.phase = 2
							dnn.remaining_work[0] = 1.0
							if dnn.node == dnn.dst_node:
								self.adjust_node_storage(dnn.DNN_size, [dnn.node], 1)
								self.service_downtime += dnn.result_delay
								self.finished_DNN_count += 1
							else:
								self.deploy_DNN(dnn)
								unfinished_DNN.append(dnn)
						elif dnn.phase == 2:
							self.adjust_node_storage(dnn.DNN_size, [dnn.node], 1)
							self.adjust_link_state(dnn.dr_req, dnn.result_link, dnn.result_link_dir, 1)
							dnn.result_delay += (self.time - dnn.time)
							self.service_downtime += dnn.result_delay
							self.finished_DNN_count += 1
						if dnn.phase != 0:
							dnn.time = self.time
					self.DNN_lst = unfinished_DNN
			self.update_end_time()
		self.time = time
		

	def Iot_move(self):
		for dt in self.DT_lst:
			if dt.DT_id >= IOT_STATIC:
				if np.random.rand() < IOT_MOVE_PROB:
					new_dst_node = int(MEC_NODE_GRAPH[dt.IoT_node][int(np.random.randint(0, len(MEC_NODE_GRAPH[dt.IoT_node])))])
					dt.IoT_node = new_dst_node
					self.update_DT(dt.DT_id, dt.node)
					self.update_end_time()

	def user_move(self):
		for dnn in self.DNN_lst:
			if np.random.rand() < IOT_MOVE_PROB:
				new_dst_node = int(MEC_NODE_GRAPH[dnn.dst_node][int(np.random.randint(0, len(MEC_NODE_GRAPH[dnn.dst_node])))])
				dnn.dst_node = new_dst_node


	

	def step(self, action):
		candid_node = 0
		candid_link = []
		candid_link_dir = []
		reward = 0
		action_applied = False
		if self.updating:
			#self.DT_lst[self.updating_DT].reset_AoI()
			if self.time >= self.DT_lst[self.updating_DT].mig_end_time:
				self.adjust_node_storage(self.DT_lst[self.updating_DT].DT_size, [self.DT_lst[self.updating_DT].node], 1)#release old DT
				self.adjust_node_state(self.DT_lst[self.updating_DT].proc_req, [self.DT_lst[self.updating_DT].node], 1)

				candid_node = self.find_max_prior_node(action, self.DT_lst[self.updating_DT].DT_size)
				action_applied = True
				candid_link, candid_link_dir = self.find_shortest_path(self.DT_lst[self.updating_DT].dr_req,
															self.DT_lst[self.updating_DT].node, 
															candid_node, 
															self.DT_lst[self.updating_DT].DT_size)
				self.adjust_node_storage(self.DT_lst[self.updating_DT].DT_size, [candid_node], 0)#allocate new DT
				self.adjust_node_state(self.DT_lst[self.updating_DT].proc_req, [candid_node], 0)				
				self.adjust_link_state(self.DT_lst[self.updating_DT].dr_req, candid_link, candid_link_dir, 0)#allocate DT migration path

				average_train_prop_delay = self.update_DT(self.updating_DT, candid_node)#重新計算DT資訊
				
				#reward = - (self.DT_lst[self.updating_DT].update_delay + average_train_prop_delay) / 1000
				"""self.updating_DT += 1

				if self.updating_DT >= NUM_DT:
					self.updating = False
					self.updating_DT = 0"""

				mig_time = self.count_prop_delay(self.DT_lst[self.updating_DT].dr_req, self.DT_lst[self.updating_DT].DT_size, candid_link, candid_link_dir)
				self.DT_lst[self.updating_DT].mig_end_time = self.time + mig_time
				if mig_time > 0:
					resource_event = Resource_Event(
						user=self.updating_DT,
						time=self.time,
						mig_num=self.mig_num[self.updating_DT],
						size=self.DT_lst[self.updating_DT].DT_size,
						res_time=self.time + mig_time,
						proc_req=self.DT_lst[self.updating_DT].proc_req,
						dr_req=self.DT_lst[self.updating_DT].dr_req,
						link=candid_link,
						link_dir=candid_link_dir,
					)
					self.resource_event_queue.append(resource_event)
					self.add_finish_time(0, self.updating_DT, self.mig_num[self.updating_DT], [self.DT_lst[self.updating_DT].mig_end_time])
					self.mig_num[self.updating_DT] += 1

				self.update_end_time()

			"""self.count_state(self.DT_lst[self.updating_DT].dr_req, self.DT_lst[self.updating_DT].proc_req, 0)

			self.cur_state_ = copy.deepcopy(self.observation_space)"""

			if not (self.updating_DT == NUM_DT - 1):
				self.updating_DT += 1
			else:
				for dt_id in range(NUM_DT):
					remaining_mig_time = max(self.DT_lst[dt_id].mig_end_time - self.time, 0)
					self.DT_lst[dt_id].update_delay = self.count_update_delay(dt_id, remaining_mig_time)
					self.DT_lst[dt_id].data_arrival_time.append(self.time_slot * TIME_SLOT + self.DT_lst[dt_id].update_delay)
					self.DT_lst[dt_id].reset_AoI()
				self.updating_DT = 0
				self.check_data_arrival()
				self.next_event()

			
		else:
			candid_node = self.find_max_prior_node(action, DNN_SIZE)
			action_applied = True
			dnn = DNN.new_DNN(self.event.user, self.time, candid_node, self.event.dst_node, self.event.delay_cons, self.event.DT_req_lst, self.event.dr_req, self.event.proc_req)

			self.adjust_node_storage(dnn.DNN_size, [candid_node], 0)#allocate new DT
			#self.adjust_node_state(dnn.proc_req, [candid_node], 0)
			"""if self.DNN_lst[self.event.user] != None:
				print("DNN", self.event.user, "is not None, replacing it!")"""
			
			self.deploy_DNN(dnn)

			train_prop_delay, estimated_train_proc_delay, estimated_result_prop_delay = self.estimate_result_delay(dnn)

			estimated_result_delay = train_prop_delay + estimated_train_proc_delay + estimated_result_prop_delay

			self.estimated_result_delay += estimated_result_delay

			#self.service_downtime = dnn.result_delay

			self.estimated_acc = dnn.acc

			self.DNN_lst.append(dnn)

			self.update_end_time()

			reward = (self.get_reward(1) * 8 - (estimated_result_delay - self.event.delay_cons) / 1000 * 2) / 10

			#reward = self.get_reward(1)

			#reward = (estimated_result_delay - self.event.delay_cons) / 1000

			self.next_event()


			#self.count_state(self.event.dr_req, self.event.proc_req, 1)
			
			#self.cur_state_ = copy.deepcopy(self.observation_space)

			#self.pre_user = dnn

		'''if not self.updating or (self.updating_DT == NUM_DT - 1):
			self.next_event()
			self.update_end_time()
			self.finish_event()'''

			

		"""if self.updating:
			self.count_state(self.DT_lst[self.updating_DT].dr_req, self.DT_lst[self.updating_DT].proc_req, 0)
		else:
			self.count_state(self.event.dr_req, self.event.proc_req, 1)
		
		self.next_state = copy.deepcopy(self.observation_space)"""

		#print(f"Time: {self.time}, Updating: {self.updating}, Updating_DT: {self.updating_DT}, Reward: {reward}")

		return reward, action_applied

	
	def get_update_delay(self, DT_id):
		return self.DT_lst[DT_id].update_delay

	def get_average_AoI(self, DT_id):
		return self.DT_lst[DT_id].AoI

	def get_average_AoI_by_step(self, DT_id):
			return self.average_AoI[DT_id] / self.req_count[DT_id]

	def get_reward(self, mode, id = -1):
		if mode == 0:
			"""if self.DT_lst[self.updating_DT - 1].AoI == 0:
				reward = - (self.DT_lst[self.updating_DT - 1].update_delay) / 1000
			else:
				reward = - (self.DT_lst[self.updating_DT - 1].AoI) / 1000"""
			if self.DT_lst[id].req_count == 0:
				reward = - (self.DT_lst[id].update_delay) / 1000
			else:
				reward = - (self.DT_lst[id].AoI+ self.DT_lst[id].update_delay) / 2 / 1000 # 

			#reward = - (self.DT_lst[self.updating_DT - 1].update_delay) / 1000

			#reward = - (self.DT_lst[self.updating_DT - 1].AoI + self.DT_lst[self.updating_DT - 1].update_delay) / 1000
		else:
			#reward = - ((self.DNN_lst[-1].result_delay - self.DNN_lst[-1].delay_cons) / 1000) + self.DNN_lst[-1].acc * 5

			#reward = self.DNN_lst[-1].acc

			reward = -self.DNN_lst[-1].AoI / 1000
		return reward

	def get_state(self, mode):
		residual_link_state = []
		if mode == 0:
			dr_req = self.DT_lst[self.updating_DT].dr_req
			proc_req = self.DT_lst[self.updating_DT].proc_req
		else:
			dr_req = self.event.dr_req
			proc_req = self.event.proc_req

		for link in range(MAX_LINK):
			for dir in range(2):
				data_rate = (dr_req / (self.link_state[link][dir] + dr_req))
				residual_link_state.append(data_rate)

		residual_CPU_state = []

		for node in range(MAX_NODES):
			proc_rate = (proc_req / (self.node_state[node] + proc_req))
			residual_CPU_state.append(proc_rate)

		if mode == 0:
			"""DT_usage = 0
			
			for i in range(self.users):
				if self.DNN_lst[i] is not None:
					if self.DNN_lst[i].DT_req_lst[self.updating_DT]:
						DT_usage += 1
	
			dnn_demand_distribution = [0.0 for _ in range(MAX_NODES)]
	
			affected_count = 0
	
			for dnn in self.DNN_lst:
				if dnn is None:
					continue
				if not dnn.DT_req_lst[self.updating_DT]:
					continue
	
				dnn_demand_distribution[dnn.node] += 1.0
	
				affected_count += 1
	
			if affected_count > 0:
				dnn_demand_distribution = [count / affected_count for count in dnn_demand_distribution]
			self.observation_space = np.array([residual_CPU_state + residual_link_state + [self.DT_lst[self.updating_DT].IoT_node / (MAX_NODES - 1), self.DT_lst[self.updating_DT].node / (MAX_NODES - 1), DT_usage / self.users] + dnn_demand_distribution])"""
			self.observation_space = np.array([residual_CPU_state + residual_link_state + 
									  [self.DT_lst[self.updating_DT].IoT_node, self.DT_lst[self.updating_DT].node, self.DT_lst[self.updating_DT].DT_size, self.DT_lst[self.updating_DT].update_size]])

		else:
			"""dt_node_lst = []
			data_lst = []
			AoI_lst = []
			for dt in self.DT_lst:
				dt_node = -1
				data = -1
				AoI = -1
				if self.event.DT_req_lst[dt.DT_id]:
					dt_node = dt.node / (MAX_NODES - 1)
					data = dt.train_data_size / MAX_DT_SIZE
					for i in range(self.time_slot + 1):
						if self.time >= dt.data_arrival_time[self.time_slot - i]:
							AoI = (self.time - (self.time_slot - i) * TIME_SLOT) / TIME_SLOT
							break
				dt_node_lst.append(dt_node)
				data_lst.append(data)
				AoI_lst.append(AoI)

			self.observation_space = np.array([residual_CPU_state + residual_link_state + [self.event.dst_node / (MAX_NODES - 1), max(AoI_lst) / TIME_SLOT] + dt_node_lst])"""
			dt_node_lst = []
			AoI_lst = []
			train_data_size = 0
			for dt in self.DT_lst:
				AoI = -1
				dt_node_lst.append(dt.node)
				if self.event.DT_req_lst[dt.DT_id]:
					train_data_size += dt.train_data_size / MAX_DT_SIZE
					for i in range(self.time_slot + 1):
						if self.time >= dt.data_arrival_time[self.time_slot - i]:
							AoI = (self.time - (self.time_slot - i) * TIME_SLOT) / TIME_SLOT
							break
				AoI_lst.append(AoI)
			self.observation_space = np.array([residual_CPU_state + residual_link_state + [self.event.dst_node, max(AoI_lst) / TIME_SLOT, train_data_size] + dt_node_lst + self.event.DT_req_lst])
	
		return np.asarray(self.observation_space, dtype=np.float32)