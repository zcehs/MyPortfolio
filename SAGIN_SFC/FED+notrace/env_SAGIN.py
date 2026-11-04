import numpy as np
import copy
from typing import List
from event import Event
#import satellite_locate.satellite_locate as satlocate
from config import (
	SPACE_STORAGE,
	AIR_STORAGE,
	GROUND_STORAGE,
	SPACE_COMPUTING,
	AIR_COMPUTING,
	GROUND_COMPUTING,
	# SPACE_BANDWIDTH,
	# AIR_BANDWIDTH,
	# GROUND_BANDWIDTH,
	MIN_RESIDENT_TIME,
	MAX_RESIDENT_TIME,
	PLC_PROBILITY,
 
	E2E_DELAY_MEAN,
	MSD_MEAN,
	MAX_SFC_TYPE,
	# MIN_SFC_DELAY,
	MIN_SFC_PACKETSIZE,

	# PERCENT_OF_LIMIT,
	# PERCENT_OF_USER_DATA,
	MIG_DR_REQ,
	MAX_WIRELESS_BANDWIDTH,

	# Event model
	MIN_SEED,
	MAX_SEED,

	# Netwrok model
	NUM_SPACE_NODES,
	NUM_AIR_NODES,
	NUM_GROUND_NODES,
	MAX_NODES,

	MEC_NODE_GRAPH, MEC_LINK_GRAPH, LINK_PROPAGATION , MAX_LINK, 
 	# SHARE_LINK, 
  	LINK_ARR, 
   	EDGE_TYPE,
	NUM_SPACE_LINKS,
	NUM_AIR_LINKS,
	NUM_GROUND_LINKS,
)
import pprint

if NUM_SPACE_LINKS + NUM_AIR_LINKS + NUM_GROUND_LINKS != MAX_LINK:
    print("Link num error")
    exit(1)
 

#####################  hyper parameters  ####################

#VNF storage requirement list 30~50(MB) 5種VNF
# VNF_lst = [ 30, 35, 40, 45, 50 ] 
VNF_lst = [ 2*10**9, 2*10**9, 4*10**9, 4*10**9 ] # 0.2~0.4 GB = 200~400 MB = 1600~3200 Mbit


#Data rate requirement list 5~9(Mbps) 3種data rate requirement	
# SFC_dr_lst = [ 5, 7, 9 ] # 5~9 Mbps
SFC_dr_lst = [ 5 *10**6, 10 *10**6 ] # 5~10 Mbps

# The data that SFC produce per second 457.5Mb 915Mb => 93, 187Mb
SFC_produce_data_lst = [ 250 *10**6, 500 *10**6 ]

SFC_size_lst = [ 2, 2 ] #個別SFC的長度

#SFC packetsize unit:Kbit
SFC_packetSize_lst = [ 8 * MIN_SFC_PACKETSIZE, 8 * MIN_SFC_PACKETSIZE ] #512Mb
SFC_lst = [ [VNF_lst[0], VNF_lst[1]],
            [VNF_lst[2], VNF_lst[3]] ] #VNF_lst中的數值為storage requirement


class ENV(object):
	def __init__(self, MAX_USER):
		self.MAX_USER = MAX_USER
		self.avg_dwell_time = np.random.uniform(MIN_SEED, MAX_SEED, self.MAX_USER).tolist()
		self.observation_space = np.array([[0] * (MAX_NODES + 2 * MAX_LINK + 3 + 2)])   # [residual storage(14), residual bandwidth(23), mig_event(3)]
		self.a_dim = MAX_NODES	                        # Action output number
		self.a_bound = 1 								# f = (0,1]	output range
		self.episode_complete = 0
		# self.MAX_NODE = MAX_NODES
		# self.MAX_LINK = MAX_LINK
		# self.node_s = node_s
		# self.node_a = node_a
		# self.node_g = node_g
		self.Node_storage = []							# All of the nodes storage resource
		# self.wireless_link_state = []
		self.Node_users = [[ 0 for i in range(self.MAX_USER) ] for j in range(MAX_NODES)]					# The user number of all nodes
		self.Link_users = [[ [0 for i in range(self.MAX_USER)],[0 for i in range(self.MAX_USER)] ] for i in range(MAX_LINK)]     # The user number of all links
		self.event_queue = []
		self.past_user_lst = []
		self.user_data_lst = []

		# Used for output
		self.total_mig_time = 0
		self.total_E2E_delay = 0
		self.times_of_mig = 0

		# times of failed episode 
		self.p_episode_fail = 0
		self.m_episode_fail = 0

		# times of failed migration
		self.migration_fail = 0

		self.max_exist_user = 0

		# Reset
		self.placement_event_SFC = [0] * 3

		self.migration_event_SFC = [0] * 3

		self.placement_event_SFC_sucess = [0] * 3

		self.migration_event_SFC_sucess = [0] * 3

		self.no_need_migration = 0

		self.Node_Utilization = [0 for i in range(MAX_NODES)]
		self.node_s_used = 0
		self.node_a_used = 0
		self.node_g_used = 0
		self.node_used = 0
		self.Link_Utilization = [[0 for i in range(2)] for j in range(MAX_LINK)] # [up, down]
		
		self.check_result_SFC_false = 0
		self.check_result_SFC_E2E_false = 0
		self.check_result_SFC_dr_false = 0
		self.check_result_SFC_mig_false = 0

		self.check_result_MIG_false = 0
		self.check_result_MIG_E2E_false = 0
		self.check_result_MIG_dr_false = 0
		self.check_result_MIG_mig_false = 0
		self.service_time = 0

		self.Placement_Down_Time = 0
		self.Migration_Down_Time = 0
		self.response_time = 0
		self.average_migration_time = 0
		self.current_SFC_index = 0
		self.target_node = 0
		# current keep migration number lst
		self.cur_mig_times_lst = [-1 for i in range(self.MAX_USER)]
		
		# max keep migration number lst
		self.max_mig_times_lst = [-1 for i in range(self.MAX_USER)]
		self.log_data_lst = []

	# start->step->step...->reset(start to next episode)->step->step->...
	def reset(self):

		self.event_queue: List[Event] = []
		# self.res_time_lst = [0 for i in range(self.MAX_USER)]
		# Migration event = [time, user, res_time, src_node, dst_node, SFC type]
		# Placement event = [time, user, res_time, -1, dst_node, SFC type]
  		# Resource(SFC & migration path) event = [time, user, res_time, dst_node, SFC type, mig_path, mig_path_dir, mig_time, residual_VNF]
		for i in range(self.MAX_USER):
			event = Event.new_placement(user=i, avg_dwell_time=self.avg_dwell_time[i])
			self.event_queue.append(event)

		# Choose the most recent time
		self.event = self.event_queue[0]
		for ev in self.event_queue:
			if ev.time < self.event.time:
				self.event = ev
		self.event_queue.remove(self.event)

		# Reset the record of the most recently SFC allocation(nodes, links) of ith user.
		self.past_user_lst = [ [[], [], [], -1, -1, -1] for j in range(self.MAX_USER)] # candid_node, candid_link, candid_link_dir, SFC_type, dst_node, E2E_delay

		# Reset the data that SFC produce per second for each user
		self.user_data_lst = [0 for i in range(self.MAX_USER)]

		# Reset all nodes storage resources
		# self.Node_storage = [MAX_STORAGE] * MAX_NODE
		self.Node_storage = [SPACE_STORAGE]*NUM_SPACE_NODES + [AIR_STORAGE]*NUM_AIR_NODES + [GROUND_STORAGE]*NUM_GROUND_NODES
		self.Max_Node_storage = copy.deepcopy(self.Node_storage)

		# Reset all wireless link bandwidth resources
		# self.wireless_link_state = [0 for i in range(MAX_NODE)]

		# Reset the user number of all nodes
		self.Node_users = [[ 0 for i in range(self.MAX_USER) ] for j in range(MAX_NODES)]

		# Reset the user number of all links
		self.Link_users = [[ [0 for i in range(self.MAX_USER)],[0 for i in range(self.MAX_USER)] ] for i in range(MAX_LINK)]

		# Reset the number of episode
		self.episode_complete = 0

		# Reset the total E2E delay
		self.total_E2E_delay = 0

		# Reset the total migration time
		self.total_mig_time = 0

		# Reset times of migration
		self.times_of_mig = 0

		# Reset of episode failed
		self.p_episode_fail = 0
		self.m_episode_fail = 0

		# Reset times of failed migration
		self.migration_fail = 0

		self.max_exist_user = 0

		# Reset
		self.placement_event_SFC = [0] * 3

		self.migration_event_SFC = [0] * 3

		self.placement_event_SFC_sucess = [0] * 3

		self.migration_event_SFC_sucess = [0] * 3

		self.no_need_migration = 0

		self.Node_Utilization = [0 for i in range(MAX_NODES)]
		self.node_s_used = 0
		self.node_a_used = 0
		self.node_g_used = 0
		self.node_used = 0
		self.Link_Utilization = [[0 for i in range(2)] for j in range(MAX_LINK)] # [up, down]

		self.check_result_SFC_false = 0
		self.check_result_SFC_E2E_false = 0
		self.check_result_SFC_dr_false = 0
		self.check_result_SFC_mig_false = 0

		self.check_result_MIG_false = 0
		self.check_result_MIG_E2E_false = 0
		self.check_result_MIG_dr_false = 0
		self.check_result_MIG_mig_false = 0

		self.Placement_Down_Time = 0
		self.Migration_Down_Time = 0
		self.response_time = 0
		self.average_migration_time = 0

		user = self.event.user
		res_time = self.event.res_time
		dst_node = self.event.dst_node
		SFC_type = self.event.SFC_type

		# State = [residual storage(MAX_NODES), residual bandwidth(MAX_LINK*2), event(3)] 
		node_state = [0 for i in range(MAX_NODES)]
		link_state = [[0 for j in range(2)]for i in range(MAX_LINK)]
		self.link_state = copy.deepcopy(link_state)
		self.node_state = copy.deepcopy(node_state)

		residual_link_state = []
		self.Max_link_state = []
		# HAP到衛星、衛星到HAP、地面到衛星、衛星到地面、HAP之間、地面到HAP、HAP到地面、地面之間
		data_rate = [[1.017, 1.035], [5.287, 5.448], [1.599, 1.599], [1.584, 1.583], [10, 10] ]
		for i in range(MAX_LINK):
			for j in range(len(EDGE_TYPE)):
				if i in EDGE_TYPE[j]:
					residual_link_state.append(data_rate[j][0]) #上
					residual_link_state.append(data_rate[j][1]) #下
					self.Max_link_state.append([data_rate[j][0]* 10**9, data_rate[j][1]* 10**9])
		# self.Max_link_state = np.array(residual_link_state, dtype="int64")
		# for i in range(len(self.Max_link_state)):
		# 	self.Max_link_state[i] *= 10**9
		# print(self.Max_link_state)

		residual_CPU_state = [SPACE_COMPUTING/10**9 for i in range(NUM_SPACE_NODES)] + [AIR_COMPUTING/10**9 for i in range(NUM_AIR_NODES)] + [GROUND_COMPUTING/10**9 for i in range(NUM_GROUND_NODES)]
		self.Max_CPU_state = [SPACE_COMPUTING for i in range(NUM_SPACE_NODES)] + [AIR_COMPUTING for i in range(NUM_AIR_NODES)] + [GROUND_COMPUTING for i in range(NUM_GROUND_NODES)]
		#link_state = [0 for i in range(2 * MAX_LINK)]
		self.past_user_E2E_delay_req = [-1 for i in range(self.MAX_USER)]
		self.E2E_delay_req = np.random.normal(E2E_DELAY_MEAN[SFC_type])
		self.past_user_E2E_delay_req[user] = self.E2E_delay_req
		self.SD_req = self.E2E_delay_req
		self.M_req = np.random.normal(MSD_MEAN[SFC_type])
		# else:
		# 	self.SD_req = np.random.normal(MSD_MEAN[SFC_type])
		self.observation_space = np.array([ residual_CPU_state + residual_link_state + [res_time, dst_node, SFC_type] + [self.SD_req , self.E2E_delay_req ]])
		
		self.service_time = 0

		# current keep migration number lst
		self.cur_mig_times_lst = [-1 for i in range(self.MAX_USER)]
		
		# max keep migration number lst
		self.max_mig_times_lst = [-1 for i in range(self.MAX_USER)]
		self.log_data_lst = []
		self.current_SFC_index = 0
		self.target_node = 0
		self.log_node_vnf = [0 for i in range(MAX_NODES)]
		self.space_no_migrate = 0
		self.air_no_migrate = 0
		self.ground_no_migrate = 0
		self.space_migrate = 0
		self.air_migrate = 0
  
		self.E2E_delay_violate = 0
		self.PSD_violate = 0
		self.MSD_violate = 0
		self.SFC_type_count = [ [0,0], [0,0], [0,0] ] # [ space[type1, type2], air[type1, type2], ground[type1, type2] ]
		# self.SFC_type_count = [ [0,0,0], [0,0,0] ]
		self.backup_s = self.observation_space
		self.no_change = 0
		self.bigger_E2E_delay = 0
		self.diff_E2E_delay = 0
		self.data_rate_violate = 0
		self.affects_other_SFC = 0
		self.avg_env_E2E = 0
		return self.observation_space

	def find_two_hub_node(self, dst_node):

		two_hub_lst = [dst_node]

		for one_hub in MEC_NODE_GRAPH[dst_node]:
			if(one_hub not in two_hub_lst):
				two_hub_lst.append(one_hub)

			for two_hub in MEC_NODE_GRAPH[one_hub]:
				if(two_hub not in two_hub_lst):
					two_hub_lst.append(two_hub)

		return two_hub_lst

	def find_three_hub_node(self, dst_node):

		three_hub_lst = [dst_node]

		for one_hub in MEC_NODE_GRAPH[dst_node]:
			if(one_hub not in three_hub_lst):
				three_hub_lst.append(one_hub)

			for two_hub in MEC_NODE_GRAPH[one_hub]:
				if(two_hub not in three_hub_lst):
					three_hub_lst.append(two_hub)

				for three_hub in MEC_NODE_GRAPH[two_hub]:
					if(three_hub not in three_hub_lst):
						three_hub_lst.append(three_hub)

		return three_hub_lst

	# Find the maximum priority node in action
	def find_max_prior_node(self, temp_link_state, temp_node_state, temp_node_storage, action, start_node, index):		
		c_node = -1
		max_prior = -1
		c_path = []
		min_E2E_delay = 0
		# max_node_storage = 0

		for j in range(self.a_dim):
			
			if(temp_node_storage[j] < SFC_lst[self.event.SFC_type][index]):
				continue

			tmp_path, tmp_dir = self.find_shortest_path(temp_link_state, start_node, j, SFC_packetSize_lst[self.event.SFC_type])
			pre_temp_link_state = copy.deepcopy(temp_link_state)
			pre_temp_link_state = self.adjust_link_state(self.event.SFC_type, 0, pre_temp_link_state, [tmp_path], [tmp_dir], 1)

			pre_temp_node_state = copy.deepcopy(temp_node_state)
			pre_temp_node_state = self.adjust_node_state(self.event.SFC_type, pre_temp_node_state, [j], 1)
			# print('find')
			temp_E2E_delay = self.count_E2E_delay(pre_temp_node_state, pre_temp_link_state, [j], [tmp_path], [tmp_dir],self.event.SFC_type)
			# print('max')
			if(action[0][j] > max_prior):
				c_node = j
				c_path = tmp_path
				c_path_dir = tmp_dir

				max_prior = action[0][j]
				min_E2E_delay = temp_E2E_delay

			elif(action[0][j] == max_prior and min_E2E_delay > temp_E2E_delay):
				c_node = j
				c_path = tmp_path
				c_path_dir = tmp_dir

				min_E2E_delay = temp_E2E_delay
		
		"""print(f'storage req = {SFC_lst[self.event.SFC_type][index]}')
		print(np.array(temp_node_storage)/10**9)	
		print(action)
		#exit()"""

		if(c_node == -1):
			# return -1, -1, -1
			print(f'action = {action[0]}')
			print(f'storage req = {SFC_lst[self.event.SFC_type][index]}')
			print(np.array(temp_node_storage)/10**9)	
			print("Find max node error.")
			print(action)
			exit()

		
		return c_node, c_path, c_path_dir

	# Dijstra algorithmn
	def find_shortest_path(self, link_state, src, dst, packetSize):
		inf = 99999999999.99999

		# Record path
		path_ = [[[] for index1 in range(MAX_NODES)] for index2 in range(MAX_NODES)]
		dir_ = [[[] for index1 in range(MAX_NODES)] for index2 in range(MAX_NODES)]

		# Init link comsumption
		edge_comsumption = np.zeros((MAX_NODES,MAX_NODES),dtype = float )
		for i in range(MAX_NODES):
			for j in range(MAX_NODES):
				if i == j:
					edge_comsumption[i][j] = 0
					path_[i][j].append(-1)
				elif(j in MEC_NODE_GRAPH[i]):
					# MEC i connect to MEC j with link MEC_LINK_GRAPH[i][MEC_NODE_GRAPH[i].index(j)]
					# transmission_dalay = packetSize / (MAX_BANDWIDTH / (temp_link_user[MEC_LINK_GRAPH[i][MEC_NODE_GRAPH[i].index(j)]] +1 ))
					propagation_dalay = LINK_PROPAGATION[MEC_LINK_GRAPH[i][MEC_NODE_GRAPH[i].index(j)]]
					edge_comsumption[i][j] = propagation_dalay # + transmission_dalay
					path_[i][j].append(MEC_LINK_GRAPH[i][MEC_NODE_GRAPH[i].index(j)])
					if(i < j):
						dir_[i][j].append(0)
					elif(i > j):
						dir_[i][j].append(1) 
				else:
					edge_comsumption[i][j] = inf
					path_[i][j].append(-1)
        
		# Init distence
		dis = np.zeros(MAX_NODES,dtype=float) # dist[i] : The edge comsumption of src to node i
		for i in range(MAX_NODES):
			if(edge_comsumption[src][i] != inf and edge_comsumption[src][i]!= 0):
				i_link = MEC_LINK_GRAPH[src][MEC_NODE_GRAPH[src].index(i)]
				if(src < i):
					data_rate = self.Max_link_state[i_link][0] * SFC_dr_lst[self.event.SFC_type] / (link_state[i_link][0] + SFC_dr_lst[self.event.SFC_type])
					
					transmission_dalay = packetSize / data_rate
					dis[i] = edge_comsumption[src][i] + transmission_dalay
				elif(src > i):
					date_rate =  self.Max_link_state[i_link][1] * SFC_dr_lst[self.event.SFC_type] / (link_state[i_link][1] + SFC_dr_lst[self.event.SFC_type])
					
					transmission_dalay = packetSize / date_rate
					dis[i] = edge_comsumption[src][i] + transmission_dalay
			else:
				dis[i] = edge_comsumption[src][i]

        # Init check
		check = np.zeros(MAX_NODES, dtype=int)
		check[src] = 1
		# Dijkstra algorithmn
		for i in range(MAX_NODES-1):
			# print(f'Dijkstra{i}')
			min = inf
			for j in range(MAX_NODES):
				if check[j] ==0 and dis[j]!=inf and dis[j]<min:
					min = dis[j]
					u = j
			check[u] = 1
			for v in range(MAX_NODES):
				if(edge_comsumption[u][v] < inf and edge_comsumption[u][v] != 0):
					i_link = MEC_LINK_GRAPH[u][MEC_NODE_GRAPH[u].index(v)]
					if(u < v):
						data_rate = self.Max_link_state[i_link][0] * SFC_dr_lst[self.event.SFC_type] / (link_state[i_link][0] + SFC_dr_lst[self.event.SFC_type])
						
						transmission_dalay = packetSize / data_rate
						if(dis[v] > dis[u] + edge_comsumption[u][v] + transmission_dalay):
							path_[src][v] = path_[src][u] + path_[u][v] 
							dir_[src][v] = dir_[src][u] + dir_[u][v]					
							dis[v] = dis[u] + edge_comsumption[u][v] + transmission_dalay

					elif(u > v):
						data_rate = self.Max_link_state[i_link][1] * SFC_dr_lst[self.event.SFC_type] / (link_state[i_link][1] + SFC_dr_lst[self.event.SFC_type])

						transmission_dalay = packetSize / data_rate
						if(dis[v] > dis[u] + edge_comsumption[u][v] + transmission_dalay):
							path_[src][v] = path_[src][u] + path_[u][v]
							dir_[src][v] = dir_[src][u] + dir_[u][v]	
							dis[v] = dis[u] + edge_comsumption[u][v] + transmission_dalay

		dst_path = []
		for link in path_[src][dst]:
			if(link != -1):
				dst_path.append(link)

		dst_dir = []
		for link in dir_[src][dst]:
			dst_dir.append(link)

		return dst_path, dst_dir

    # Count propagation delay of virtual link
	def count_E2E_delay(self, node_state, link_state, candid_node, candid_link, candid_link_dir, SFC_type):

		prop_delay = 0
		trans_delay = 0
		proc_delay = 0

		# print()
		if(candid_node):
			for node in candid_node:
				if(node_state[node] == 0):
					print("node state error 480")
					exit()
				proc_rate = self.Max_CPU_state[node] * SFC_dr_lst[SFC_type] / node_state[node]

				# proc_delay = proc_delay + SFC_packetSize_lst[SFC_type] / proc_rate # Mbit / Gbps (ms)
				proc_delay = proc_delay + SFC_packetSize_lst[SFC_type] / proc_rate*1000 # Mbit / Gbps (ms)
		
		if(candid_link):

			for v_link_index in range(len(candid_link)):
				min_data_rate = 9999 *10**10
				v_link = candid_link[v_link_index]

				for link_index in range(len(v_link)):
					link = v_link[link_index]
					dir = candid_link_dir[v_link_index][link_index]
					data_rate = self.Max_link_state[link][dir] * SFC_dr_lst[SFC_type] / link_state[link][dir]

					if(data_rate < min_data_rate):
						min_data_rate = data_rate
					prop_delay = prop_delay + LINK_PROPAGATION[link] # ms				

				if(len(v_link) != 0):
					# trans_delay = trans_delay + SFC_packetSize_lst[SFC_type] /  min_data_rate # Mbit / Gbps (ms)
					trans_delay = trans_delay + SFC_packetSize_lst[SFC_type] /  (min_data_rate*10**6) # Mbit / Gbps (ms)

		# return prop_delay + trans_delay*1000 + proc_delay*1000  # ms
		return prop_delay + trans_delay + proc_delay  # ms
    
	# Count migration time 
	def count_mig_time(self, SFC_type, residual_user_data, temp_link_state, mig_path, mig_dir):   

		mig_time_lst = []
				
		if(len(residual_user_data) != len(mig_path)):
			print("res_SFC != mig_path!")
			exit(1)

		for v_index in range(len(residual_user_data)):
			
			mig_time = 0
			min_data_rate = 9999 *10**10

			# Count migration time
			for link_index in range(len(mig_path[v_index])):
				link = mig_path[v_index][link_index]
				dir = mig_dir[v_index][link_index]	

				# VNF size is Byte	
				data_rate = self.Max_link_state[link][dir] * MIG_DR_REQ[SFC_type] / temp_link_state[link][dir]
				if(min_data_rate > data_rate):
					min_data_rate = data_rate

				mig_time = mig_time + (LINK_PROPAGATION[link])
				# print("link = ", link)
				# print("dir = ",dir)
				# print("date rate = ", data_rate)
				# print()
			if(len(mig_path[v_index]) != 0):
				mig_time = mig_time + (residual_user_data[v_index] / min_data_rate ) 
				# mig_time = mig_time + (residual_user_data[v_index] / min_data_rate ) # Mbit / Gbps

			# Find the max migration time
			mig_time_lst.append(mig_time)

		return mig_time_lst # ms
	
	# Count total usage from all user of each link 
	def count_link_user(self, temp_link_user, i_link, direct):       # direct=0 : n0 to n1     direct=1 : n1 to n0
		
		sum = 0
		for num in temp_link_user[i_link][direct]:
			
			sum = sum + num

		return sum

	# Count total response time for all user
	def count_E2E(self):
		
		respond_time = 0
		respond_time_count = 0
		for user in range(self.MAX_USER):
			for ev in self.event_queue:
				if ev.is_migration() and ev.user == user:
					respond_time += self.past_user_lst[user][5]
					respond_time_count += 1
					if(self.past_user_lst[user][5] == -1):
						print("Count SRT error!")
						exit(1)
					break
		
		if(respond_time_count == 0):
			return 0

		return respond_time / respond_time_count

	def cal_avg_env_E2E(self, temp_node_state, temp_link_state):
		
		sum_evn_E2E = 0
		total_evn_SFC_count = 0
		for user in range(self.MAX_USER):
			for ev in self.event_queue:
				if ev.is_resource() and ev.user == user:
					# respond_time += self.past_user_lst[user][5]
					# respond_time_count += 1
					# if(self.past_user_lst[user][5] == -1):
					# 	print("Count SRT error!")
					# 	exit(1)
					# break
					past_node = self.past_user_lst[user][0]
					past_link = self.past_user_lst[user][1]
					past_link_dir = self.past_user_lst[user][2]
					past_SFC_type = int(self.past_user_lst[user][3])
					# past_user_dst = int(self.past_user_lst[user][4])
					past_E2E_delay = self.count_E2E_delay(temp_node_state, temp_link_state, past_node, past_link, past_link_dir, past_SFC_type)
					self.past_user_lst[user][5] = past_E2E_delay
					sum_evn_E2E += past_E2E_delay
					total_evn_SFC_count += 1
		if(total_evn_SFC_count == 0):
			return 0

		return sum_evn_E2E / total_evn_SFC_count

	# Count total migration time for all user
	def count_MDT(self):

		total_migration_time = 0
		total_migration_time_count = 0

		for user in range(self.MAX_USER):
			max_migration_time = 0
			
			for ev in self.event_queue:
				if(ev.is_resource() and ev.user == user): # No plc or mig event in event queue
					if(max_migration_time < ev.mig_time):
						max_migration_time = ev.mig_time

						if(ev.mig_time == 0):
							print("count total migration time ev.mig_time == 0 !")
							exit(1)

				elif ev.is_migration() and ev.user == user: # mig event
					total_migration_time_count += 1

			if(max_migration_time != 0):
				total_migration_time =	total_migration_time + max_migration_time + self.past_user_lst[user][5]
				total_migration_time_count += 1

		if(total_migration_time_count == 0):
			return 0
		else:
			return total_migration_time / total_migration_time_count
	
	# Check the result after the impact of the SFC event
	def check_result_after_SFC_impact(self, temp_node_state, temp_link_state, candid_link, candid_link_dir, operator_):	
		
		check_result = True

		# 1. mig time
		count = 0
		temp_event_queue = copy.deepcopy(self.event_queue)
		for ev in self.event_queue:
			if ev.is_resource():
				# Resource(SFC & migration path) event = [time, user, res_time, dst_node, SFC type, mig_path, mig_path_dir, mig_time, residual_VNF]
				temp_time = ev.time
				temp_res_time = ev.res_time * 1000      # residual res time
				temp_SFC_type = ev.SFC_type
				temp_mig_path = ev.mig_path
				temp_mig_path_dir = ev.mig_path_dir
				temp_mig_time = ev.mig_time      # residual mig time
				temp_residual_user_data = ev.residual_VNF  # residual user data


				already_mig_time = self.event.time - (temp_time - temp_mig_time)
				temp_res_time = temp_res_time - already_mig_time

				temp_residual_user_data = temp_residual_user_data * (1 -(already_mig_time / temp_mig_time))

				new_mig_time = self.count_mig_time(temp_SFC_type, [temp_residual_user_data], temp_link_state, [temp_mig_path], [temp_mig_path_dir])[0]

				if(temp_res_time < new_mig_time):
					self.check_result_SFC_false += 1
					self.check_result_SFC_mig_false += 1
					print("check result mig time false 658")
					print()
					exit(1)

				temp_event_queue[count][0] = self.event.time + new_mig_time
				temp_event_queue[count][2] = temp_res_time
				temp_event_queue[count][7] = new_mig_time
				temp_event_queue[count][8] = temp_residual_user_data

			count = count + 1

		self.event_queue = copy.deepcopy(temp_event_queue)

		for user in range(self.MAX_USER):
			if(self.past_user_lst[user] != [[],[],[],-1,-1,-1] and user != self.event.user):
				past_node = self.past_user_lst[user][0]
				past_link = self.past_user_lst[user][1]
				past_link_dir = self.past_user_lst[user][2]
				past_SFC_type = int(self.past_user_lst[user][3])
				past_user_dst = int(self.past_user_lst[user][4])
				# print('check sfc')
				past_E2E_delay = self.count_E2E_delay(temp_node_state, temp_link_state, past_node, past_link, past_link_dir, past_SFC_type)
				# print('sfc')		
				wireless_delay = SFC_packetSize_lst[past_SFC_type] / MAX_WIRELESS_BANDWIDTH
						
				past_E2E_delay = past_E2E_delay + wireless_delay
					
				self.past_user_lst[user][5] = past_E2E_delay

				if(past_E2E_delay > self.past_user_E2E_delay_req[user] and operator_ == 1):
					self.check_result_SFC_false += 1
					self.check_result_SFC_E2E_false += 1
						# print("E2E false")
						# print()
					check_result = False

		return check_result

	# Check the result after the impact of the mig event
	def check_result_after_MIG_impact(self, temp_node_state, temp_link_state, mig_path, mig_path_dir, operator_):
		
		check_result = True

		# 1. mig time
		count = 0
		temp_event_queue = copy.deepcopy(self.event_queue)

		for ev in self.event_queue:
			if ev.is_resource():
				# Resource(SFC & migration path) event = [time, user, res_time, dst_node, SFC type, mig_path, mig_path_dir, mig_time, residual_VNF]
				temp_time = ev.time
				temp_res_time = ev.res_time * 1000      # residual res time
				temp_SFC_type = ev.SFC_type
				temp_mig_path = ev.mig_path
				temp_mig_path_dir = ev.mig_path_dir
				temp_mig_time = ev.mig_time      # residual mig time
				temp_residual_user_data = ev.residual_VNF  # residual user data

				already_mig_time = self.event.time - (temp_time - temp_mig_time)
				temp_res_time = temp_res_time - already_mig_time

				temp_residual_user_data = temp_residual_user_data * (1 -(already_mig_time / temp_mig_time))

				new_mig_time = self.count_mig_time(temp_SFC_type, [temp_residual_user_data], temp_link_state, [temp_mig_path], [temp_mig_path_dir])[0]

				if(temp_res_time < new_mig_time):
					self.check_result_MIG_false += 1
					self.check_result_MIG_mig_false += 1
					print("check result mig time false 734")
					print()
					exit(1)

				temp_event_queue[count][0] = self.event.time + new_mig_time
				temp_event_queue[count][2] = temp_res_time
				temp_event_queue[count][7] = new_mig_time
				temp_event_queue[count][8] = temp_residual_user_data

			count = count + 1

		self.event_queue = copy.deepcopy(temp_event_queue)

		# 2. E2E delay			
		for user in range(self.MAX_USER):
			if(self.past_user_lst[user] != [[],[],[],-1,-1,-1] and user != self.event.user):
				past_node = self.past_user_lst[user][0]
				past_link = self.past_user_lst[user][1]
				past_link_dir = self.past_user_lst[user][2]
				past_SFC_type = int(self.past_user_lst[user][3])
				past_user_dst = int(self.past_user_lst[user][4])
				# print('check mig')
				past_E2E_delay = self.count_E2E_delay(temp_node_state, temp_link_state, past_node, past_link, past_link_dir, past_SFC_type)
				# print('mig')
				wireless_delay = SFC_packetSize_lst[past_SFC_type] / MAX_WIRELESS_BANDWIDTH 
						
				past_E2E_delay = past_E2E_delay + wireless_delay

				self.past_user_lst[user][5] = past_E2E_delay

				if(past_E2E_delay > self.past_user_E2E_delay_req[user] and operator_ == 1):
					self.check_result_MIG_false += 1
					self.check_result_MIG_E2E_false += 1

					# print("E2E false")
					# print()
					check_result = False

		return check_result

	# Adjust state of each node
	def adjust_node_storage(self, c_node, temp_node_storage, VNF_size):
		temp_node_storage[c_node] = temp_node_storage[c_node] - VNF_size

		return temp_node_storage

	# Adjust number of users in each node
	def adjust_node_user(self, user, candid_node, temp_node_user, operator_):
		# operator_ : 1 => add  ，: -1 => sub

		if(candid_node):
			for node in candid_node:
				temp_node_user[node][user] += operator_
		
		return temp_node_user

	# Adjust number of users in each link
	def adjust_link_user(self, user, candid_link, candid_link_dir, temp_link_user, operator_):
		# operator_ : 1 => add  ，: -1 => sub

		if(candid_link):

			for v_link_index in range(len(candid_link)):
				v_link = candid_link[v_link_index]

				for link_index in range(len(v_link)):	

					link = v_link[link_index]
					dir = candid_link_dir[v_link_index][link_index]

					temp_link_user[link][dir][user] = temp_link_user[link][dir][user] + operator_

		return temp_link_user

	# Adjust total data rate of users in each node
	def adjust_node_state(self, SFC_type, node_state, candid_node, operator_):
		
		if(candid_node):
			for node in candid_node:
				node_state[node] = node_state[node] + operator_ * SFC_dr_lst[SFC_type]
		
		return node_state

	# Adjust total data rate of users in each link
	def adjust_link_state(self, SFC_type, event_type, link_state, candid_link, candid_link_dir, operator_):

		if(candid_link):
			for v_link_index in range(len(candid_link)):
				v_link = candid_link[v_link_index]	
				for link_index in range(len(v_link)):
					link = v_link[link_index]
					dir = candid_link_dir[v_link_index][link_index]

					if(event_type == 0):
						link_state[link][dir] = link_state[link][dir] + operator_ * SFC_dr_lst[SFC_type]
					else:
						link_state[link][dir] = link_state[link][dir] + operator_ * MIG_DR_REQ[SFC_type]

		return link_state

    # Allocate resource
	def single_allocate_resource(self, user, SFC_type, event_type, node_storage, node_state, link_state, candid_node, candid_link, candid_link_dir,current_SFC_index):
		
		# evnet_type = 0 => SFC
		# event_type = 1 => migration VNF   

		###############  Allocate storage  ###############

		# Adjust node_state
		if(candid_node):
			node_storage[candid_node[0]] = node_storage[candid_node[0]] - SFC_lst[SFC_type][current_SFC_index]
			self.log_node_vnf[candid_node[0]] += 1
			if(node_storage[candid_node[0]] < 0):
				print("Node storage error!")
				exit(1)
		###############  Allocate computing  ###############

		# Adjust the number of user in all nodes

		if(candid_node):
			for node in candid_node:
				self.Node_users[node][user] = self.Node_users[node][user] + 1
				node_state[node] = node_state[node] + SFC_dr_lst[SFC_type]

		###############  Allocate bandwidth  ###############

		# Adjust the number of user in all links
		if(candid_link):

			for v_link_index in range(len(candid_link)):
				v_link = candid_link[v_link_index]

				for link_index in range(len(v_link)):

					link = v_link[link_index]
					dir = candid_link_dir[v_link_index][link_index]
					
					self.Link_users[link][dir][user] = self.Link_users[link][dir][user] + 1
					if(event_type == 0):
						link_state[link][dir] = link_state[link][dir] + SFC_dr_lst[SFC_type]
					else:
						link_state[link][dir] = link_state[link][dir] + MIG_DR_REQ[SFC_type]
					
     				# share link
					# s_link_index = -1
					# for idx in range(len(share_link)):
					# 	if link in share_link[idx]:
					# 		s_link_index = idx
					# 		break
					# if s_link_index == -1 :
					# 	self.Link_users[link][dir][user] = self.Link_users[link][dir][user] + 1
					# 	if(event_type == 0):
					# 		link_state[link][dir] = link_state[link][dir] + SFC_dr_lst[SFC_type]
					# 	else:
					# 		link_state[link][dir] = link_state[link][dir] + MIG_DR_REQ[SFC_type]
					# else:
					# 	if(event_type == 0):
					# 		dr_req =  SFC_dr_lst[SFC_type]
					# 	else:
					# 		dr_req =  MIG_DR_REQ[SFC_type]
					# 	for i in range(len(share_link[s_link_index])):
					# 		self.Link_users[share_link[s_link_index][i]][dir][user] = self.Link_users[share_link[s_link_index][i]][dir][user] + 1
					# 		link_state[share_link[s_link_index][i]][dir] = link_state[share_link[s_link_index][i]][dir] + dr_req

		return node_storage, node_state, link_state

	def allocate_resource(self, user, SFC_type, event_type, node_storage, node_state, link_state, candid_node,
								 candid_link, candid_link_dir):

		# evnet_type = 0 => SFC
		# event_type = 1 => migration VNF

		###############  Allocate storage  ###############

		# Adjust node_state
		if (candid_node):
			for index in range(SFC_size_lst[SFC_type]):
				node_storage[candid_node[index]] = node_storage[candid_node[index]] - SFC_lst[SFC_type][index]
				if(node_storage[candid_node[index]] < 0):
					print("Node storage error!")
					exit(1)
		###############  Allocate computing  ###############

		# Adjust the number of user in all nodes

		if (candid_node):
			for node in candid_node:
				self.Node_users[node][user] = self.Node_users[node][user] + 1
				node_state[node] = node_state[node] + SFC_dr_lst[SFC_type]

		###############  Allocate bandwidth  ###############

		# Adjust the number of user in all links
		if (candid_link):

			for v_link_index in range(len(candid_link)):
				v_link = candid_link[v_link_index]

				for link_index in range(len(v_link)):

					link = v_link[link_index]
					dir = candid_link_dir[v_link_index][link_index]

					self.Link_users[link][dir][user] = self.Link_users[link][dir][user] + 1
					if (event_type == 0):
						link_state[link][dir] = link_state[link][dir] + SFC_dr_lst[SFC_type]
					else:
						link_state[link][dir] = link_state[link][dir] + MIG_DR_REQ[SFC_type]
     
					# share link
					# s_link_index = -1
					# for idx in range(len(share_link)):
					# 	if link in share_link[idx]:
					# 		s_link_index = idx
					# 		break
					# if s_link_index == -1 :
					# 	self.Link_users[link][dir][user] = self.Link_users[link][dir][user] + 1
					# 	if(event_type == 0):
					# 		link_state[link][dir] = link_state[link][dir] + SFC_dr_lst[SFC_type]
					# 	else:
					# 		link_state[link][dir] = link_state[link][dir] + MIG_DR_REQ[SFC_type]
					# else:
					# 	if(event_type == 0):
					# 		dr_req =  SFC_dr_lst[SFC_type]
					# 	else:
					# 		dr_req =  MIG_DR_REQ[SFC_type]
					# 	for i in range(len(share_link[s_link_index])):
					# 		self.Link_users[share_link[s_link_index][i]][dir][user] = self.Link_users[share_link[s_link_index][i]][dir][user] + 1
					# 		link_state[share_link[s_link_index][i]][dir] = link_state[share_link[s_link_index][i]][dir] + dr_req

		return node_storage, node_state, link_state
	
 	# Release resource
	
	def release_resource(self, user, SFC_type, event_type, node_storage, node_state, link_state, candid_node, candid_link, candid_link_dir):
		#Resource(SFC & migration path) event = [time, user, res_time, dst_node, SFC type, mig_path, mig_time]

		###############  Release storage  ###############

		# Adjust node_state
		if(candid_node):
			# for index in range(SFC_size_lst[SFC_type]):
			for index in range(len(candid_node)):
				node_storage[candid_node[index]] = node_storage[candid_node[index]] + SFC_lst[SFC_type][index]
				self.log_node_vnf[candid_node[index]] -= 1
				if(node_storage[candid_node[index]] > self.Max_Node_storage[candid_node[index]]):
					print("Node storage error!!")
					exit(1)

		###############  Release computing  ###############

		# Adjust the number of user in all nodes
		if(candid_node):
			for node in candid_node:
				
				self.Node_users[node][user] = self.Node_users[node][user] - 1
				node_state[node] = node_state[node] - SFC_dr_lst[SFC_type]

				if(self.Node_users[node][user] < 0):
					print("Node_user error")
					exit()
		

			
		###############  Release bandwidth  ###############
		
		# Adjust the number of user in all links
		if(candid_link):

			for v_link_index in range(len(candid_link)):
				v_link = candid_link[v_link_index]
				
				for link_index in range(len(v_link)):

					link = v_link[link_index]
					dir = candid_link_dir[v_link_index][link_index]
					if(self.Link_users[link][dir][user] - 1 < 0):
						print("Link_user error!!")
      
					self.Link_users[link][dir][user] = self.Link_users[link][dir][user] - 1
					if(event_type == 0):
						link_state[link][dir] = link_state[link][dir] - SFC_dr_lst[SFC_type]
					else:
						link_state[link][dir] = link_state[link][dir] - MIG_DR_REQ[SFC_type]

					# share link
					# s_link_index = -1
					# for idx in range(len(share_link)):
					# 	if link in share_link[idx]:
					# 		s_link_index = idx
					# 		break
					# if s_link_index == -1 :
					# 	self.Link_users[link][dir][user] = self.Link_users[link][dir][user] - 1
					# 	if(event_type == 0):
					# 		link_state[link][dir] = link_state[link][dir] - SFC_dr_lst[SFC_type]
					# 	else:
					# 		link_state[link][dir] = link_state[link][dir] - MIG_DR_REQ[SFC_type]
					# else:
					# 	if(event_type == 0):
					# 		dr_req =  SFC_dr_lst[SFC_type]
					# 	else:
					# 		dr_req =  MIG_DR_REQ[SFC_type]
					# 	for i in range(len(share_link[s_link_index])):
					# 		self.Link_users[share_link[s_link_index][i]][dir][user] = self.Link_users[share_link[s_link_index][i]][dir][user] - 1
					# 		link_state[share_link[s_link_index][i]][dir] = link_state[share_link[s_link_index][i]][dir] - dr_req
       
					if(link_state[link][dir] < 0):
						print("link_state error!")
						exit(1)
					if(self.Link_users[link][dir][user] < 0): 					
						print("Link_user error!!")
						exit()

		return node_storage, node_state, link_state
	
	# Count max exist user 
	def count_exist_user(self):

		tmp_exist_user = 0
		for i in range(self.MAX_USER):
			if(self.past_user_lst[i] != [[],[],[],-1,-1,-1]):
				tmp_exist_user += 1

		if(self.max_exist_user < tmp_exist_user):
			self.max_exist_user = tmp_exist_user
	
	def dbm_to_w(self, dbm):
		return (10**(dbm/10)) /1000

	def count_traverse_energy(self, candid_node, candid_link, candid_link_dir, SFC_type, node_state, link_state):
		trans_energy = 0
		proc_energy = 0
		watts = []
		trans_delay = 0
		# print()
		if(candid_node):
			for node in candid_node:
				# vnf_energy =  (SFC_dr_lst[SFC_type]/self.Max_CPU_state[node]) * self.node_power[node]
				proc_rate = self.Max_CPU_state[node] * SFC_dr_lst[SFC_type] / node_state[node]
				# proc_energy_dbm = pow(10,-28) * pow(proc_rate,2) * (SFC_dr_lst[SFC_type]/1000)
				proc_energy_dbm = pow(10,-28) * pow(proc_rate,3) * (SFC_packetSize_lst[SFC_type] / (proc_rate*10**6))
				# proc_energy = pow(10,-28) * pow((proc_rate*pow(10,8)),2) * (SFC_dr_lst[SFC_type]*pow(10,8))
				
				proc_energy += self.dbm_to_w(proc_energy_dbm)
			# print(f"proc_energy_dbm:{proc_energy_dbm}, proc_energy:{proc_energy}")
		
		if(candid_link):
			# print(candid_link)
			for v_link_index in range(len(candid_link)):
				min_data_rate = 9999 *10**10
				v_link = candid_link[v_link_index]

				for link_index in range(len(v_link)):
					link = v_link[link_index]
					dir = candid_link_dir[v_link_index][link_index]
					data_rate = self.Max_link_state[link][dir] * SFC_dr_lst[SFC_type] / link_state[link][dir]
						# data_rate = self.Max_link_state[link][dir] * MIG_DR_REQ[SFC_type] / link_state[link][dir]
					phy_link = LINK_ARR[link]
					# 發射端: 衛星
					if phy_link[dir] < NUM_SPACE_NODES:
						watts.append(100)
					# 發射端: HAP
					elif phy_link[dir] < NUM_SPACE_NODES+NUM_AIR_NODES :
						watts.append(7.94)
					# 發射端: 地面之間
					elif phy_link[dir] < NUM_SPACE_NODES+NUM_AIR_NODES+NUM_GROUND_NODES and  (phy_link[dir^1]>=NUM_SPACE_NODES+NUM_AIR_NODES):
						watts.append(0.001)
					else:
						watts.append(10)
     				# # 衛星-地面 or 地面-衛星
					# if (phy_link[dir] == 0 and phy_link[dir^1]>=7) or (phy_link[dir^1] == 0 and phy_link[dir]>=7): 
					# 	watts.append(10)
					# # 衛星-HAP or HAP-衛星
					# elif (phy_link[dir] == 0 and phy_link[dir^1] in [x for x in range(1,7)]) or (phy_link[dir^1] == 0 and phy_link[dir] in [x for x in range(1,7)]): 
					# 	watts.append(8)
					# # HAP-HAP
					# elif phy_link[dir] in [1,2,3,4,5,6] and phy_link[dir^1] in [1,2,3,4,5,6]: 
					# 	watts.append(2)
					# # HAP-地面 or 地面-HAP
					# elif (phy_link[dir] in [1,2,3,4,5,6] and phy_link[dir^1]>=7) or (phy_link[dir^1] in [1,2,3,4,5,6] and phy_link[dir]>=7):
					# 	watts.append(2)
					# else: #地面之間
					# 	watts.append(0.001)

					if(data_rate < min_data_rate):
						min_data_rate = data_rate
					# prop_delay = prop_delay + (LINK_PROPAGATION[link]/1000) # ms				

				if(len(v_link) != 0):
					trans_delay = trans_delay + SFC_packetSize_lst[SFC_type] /  (min_data_rate*10**6) # s
						# trans_delay = trans_delay + (8 * residual_user_data[v_index] / min_data_rate ) # Mbit / Gbps (ms)
								
			trans_energy = trans_delay * sum(watts) #* (SFC_dr_lst[SFC_type]/SFC_packetSize_lst[SFC_type])#(dr=datasize /packetsize) # J
		# print(f'proc:{proc_energy}, trans:{trans_energy}')
		return (trans_energy + proc_energy) #* (SFC_dr_lst[SFC_type]/SFC_packetSize_lst[SFC_type])#(dr=datasize /packetsize) # J
	
	def count_mig_energy(self, mig_path, mig_dir, SFC_type, link_state , residual_user_data):
		mig_energy = 0
		mig_time = 0
		watts = []
		for v_index in range(len(residual_user_data)):
			
			# mig_time = 0
			min_data_rate = 9999 *10**10

			# Count migration time
			for link_index in range(len(mig_path[v_index])):
				link = mig_path[v_index][link_index]
				dir = mig_dir[v_index][link_index]	

				# VNF size is Byte	
				data_rate = self.Max_link_state[link][dir] * MIG_DR_REQ[SFC_type] / link_state[link][dir]
				if(min_data_rate > data_rate):
					min_data_rate = data_rate

				# 地面-衛星 10w O
				# 地面-HAP   2w o
				# 地面-地面 0.001w
				# HAP-衛星   8w O
				# HAP-HAP    2w O
				# HAP到衛星、衛星到HAP、地面到衛星、衛星到地面、HAP之間、地面到HAP、HAP到地面、地面之間
				phy_link = LINK_ARR[link]
				# 發射端: 衛星
				if phy_link[dir] < NUM_SPACE_NODES:
					watts.append(100)
				# 發射端: HAP
				elif phy_link[dir] < NUM_SPACE_NODES+NUM_AIR_NODES :
					watts.append(7.94)
				# 發射端: 地面之間
				elif phy_link[dir] < NUM_SPACE_NODES+NUM_AIR_NODES+NUM_GROUND_NODES and  (phy_link[dir^1]>=NUM_SPACE_NODES+NUM_AIR_NODES):
					watts.append(0.001)
				else:
					watts.append(10)
				# 衛星-地面 or 地面-衛星
				# if (phy_link[dir] == 0 and phy_link[dir^1]>=7) or (phy_link[dir^1] == 0 and phy_link[dir]>=7): 
				# 	watts.append(10)
				# # 衛星-HAP or HAP-衛星
				# elif (phy_link[dir] == 0 and phy_link[dir^1] in [x for x in range(1,7)]) or (phy_link[dir^1] == 0 and phy_link[dir] in [x for x in range(1,7)]): 
				# 	watts.append(8)
				# # HAP-HAP
				# elif phy_link[dir] in [1,2,3,4,5,6] and phy_link[dir^1] in [1,2,3,4,5,6]: 
				# 	watts.append(2)
				# # HAP-地面 or 地面-HAP
				# elif (phy_link[dir] in [1,2,3,4,5,6] and phy_link[dir^1]>=7) or (phy_link[dir^1] in [1,2,3,4,5,6] and phy_link[dir]>=7):
				# 	watts.append(2)
				# else: #地面之間
				# 	watts.append(0.001)
      
				# mig_time = mig_time + (LINK_PROPAGATION[link])

			if(len(mig_path[v_index]) != 0):
				mig_time = mig_time + (residual_user_data[v_index] / (min_data_rate*1000) ) # Mb / Gbps
				# mig_time = mig_time + (SFC_packetSize_lst[SFC_type] /  (min_data_rate*10**6)) # s 只傳一個packet的能耗
		mig_energy = mig_time * sum(watts)
		# print(f'mig_energy:{mig_energy}')
		return mig_energy  # J

	# Cheak whether old placement meet the requirement
	def try_old_placement(self, node_storage, node_state, link_state):
		time = self.event.time              # Time of deeling this event
		user = self.event.user              # User of this event		
		res_time = self.event.res_time      # User's residient time		
		src_node = self.event.src_node      # User's source node		
		dst_node = self.event.dst_node      # User's destination node		
		SFC_type = self.event.SFC_type      # The SFC type used by user

		past_node = copy.deepcopy(self.past_user_lst[user][0])
		past_link = copy.deepcopy(self.past_user_lst[user][1])
		past_link_dir = copy.deepcopy(self.past_user_lst[user][2])
		
		new_node = copy.deepcopy(self.past_user_lst[user][0])
		new_link = copy.deepcopy(self.past_user_lst[user][1])
		new_link_dir = copy.deepcopy(self.past_user_lst[user][2])

		First_VNF_node = past_node[0]
		First_VNF_path = past_link[0]             
		First_VNF_path_dir = past_link_dir[0]   

		Last_VNF_node = past_node[-1]
		Last_VNF_path = past_link[-1]
		Last_VNF_path_dir = past_link_dir[-1]

		# Recover link state
		node_storage, node_state, link_state = self.release_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [First_VNF_path], [First_VNF_path_dir])
		node_storage, node_state, link_state = self.release_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [Last_VNF_path], [Last_VNF_path_dir])		

		# Find new path and allocate
		First_VNF_path, First_VNF_path_dir = self.find_shortest_path(link_state, dst_node, First_VNF_node, SFC_packetSize_lst[SFC_type])
		node_storage, node_state, link_state = self.allocate_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [First_VNF_path], [First_VNF_path_dir])

		Last_VNF_path, Last_VNF_path_dir = self.find_shortest_path(link_state, Last_VNF_node, dst_node, SFC_packetSize_lst[SFC_type])
		node_storage, node_state, link_state = self.allocate_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [Last_VNF_path], [Last_VNF_path_dir])
		
		new_link[0] = First_VNF_path
		new_link[-1] = Last_VNF_path
		new_link_dir[0] = First_VNF_path_dir
		new_link_dir[-1] = Last_VNF_path_dir

		# Count E2E delay
		# print("self.Link_users", self.Link_users)
		# print('try')
		E2E_delay = self.count_E2E_delay(node_state, link_state, new_node, new_link, new_link_dir, SFC_type)
		# print('old')
		wireless_delay = SFC_packetSize_lst[SFC_type] / MAX_WIRELESS_BANDWIDTH

		E2E_delay = E2E_delay + wireless_delay

		min_data_rate = 9999 *10**10	

		for link_index in range(len(Last_VNF_path)):
			
			link = Last_VNF_path[link_index]
			Last_VNF_dir = Last_VNF_path_dir[link_index]
			Last_VNF_path_data_rate = self.Max_link_state[link][Last_VNF_dir] * SFC_dr_lst[SFC_type] / link_state[link][Last_VNF_dir]
			
			if(min_data_rate > Last_VNF_path_data_rate):
				min_data_rate = Last_VNF_path_data_rate	

		# print("allocate link state")
		# print(link_state)
		# print("allocate node state")
		# print(node_state)
		# print()
		data_rate_req = SFC_dr_lst[SFC_type]
		# print('1046')
		# check_result = self.check_result_after_SFC_impact(node_state, link_state, new_link, new_link_dir, 1)
		if(E2E_delay>self.past_user_E2E_delay_req[user]):
			self.bigger_E2E_delay += 1
			self.diff_E2E_delay += (E2E_delay-self.past_user_lst[user][5])
		if(data_rate_req > min_data_rate): 
			self.data_rate_violate += 1
		# if(not check_result):
		# 	self.affects_other_SFC += 1
		if(E2E_delay < self.past_user_E2E_delay_req[user]):  #不需重新佈署
		# if(True):
			self.past_user_lst[user][0] = copy.deepcopy(new_node)
			self.past_user_lst[user][1] = copy.deepcopy(new_link)
			self.past_user_lst[user][2] = copy.deepcopy(new_link_dir)
			self.past_user_lst[user][3] = SFC_type
			self.past_user_lst[user][4] = dst_node
			self.past_user_lst[user][5] = E2E_delay

			#################################  Loading information  ######################################
			# storage loading
			node_loading = []
			for i in range(MAX_NODES):
				node_loading.append(node_storage[i]/self.Max_Node_storage[i])

			if (user == 0):
				# log infomation
				self.log_data_lst.append(src_node)
				self.log_data_lst.append(dst_node)
				self.log_data_lst.append(new_node)
				self.log_data_lst.append(new_link)
				self.log_data_lst.append(new_link_dir)
				self.log_data_lst.append(node_loading)
				self.log_data_lst.append(E2E_delay)
				self.log_data_lst.append(0)
				self.log_data_lst.append(0)
				self.log_data_lst.append(SFC_type)
			
			event = Event.new_resource(
       									user=user,
									   	time=time+res_time,
									   	res_time=res_time,
										dst_node=dst_node,
									   	SFC_type=SFC_type,
										mig_path=[],
										mig_path_dir=[],
										mig_time=0,
										residual_VNF=0
                              )
			self.event_queue.append(event)
			self.Node_storage = node_storage
			self.node_state = node_state
			self.link_state = link_state
			return True, E2E_delay

		else: #需要重新佈署
			# release new path
			node_storage, node_state, link_state = self.release_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [new_link[0]], [new_link_dir[0]])
			node_storage, node_state, link_state = self.release_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [new_link[-1]], [new_link_dir[-1]])
			
			# allocate old path
			node_storage, node_state, link_state = self.allocate_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [past_link[0]], [past_link_dir[0]])
			node_storage, node_state, link_state = self.allocate_resource(user, SFC_type, 0, node_storage, node_state, link_state, [], [past_link[-1]], [past_link_dir[-1]])
			self.Node_storage = node_storage
			self.node_state = node_state
			self.link_state = link_state
			return False, -1
	
	def get_SFC_len(self):
		SFC_len = SFC_size_lst[self.event.SFC_type]   # Length of SFC
		return SFC_len	
	
	def random_dst_node(self, src_node:int):
		dst_node = int(MEC_NODE_GRAPH[src_node][int(np.random.randint(0, len(MEC_NODE_GRAPH[src_node])))])
		while dst_node < (NUM_SPACE_NODES+NUM_AIR_NODES):
			dst_node = int(MEC_NODE_GRAPH[src_node][int(np.random.randint(0, len(MEC_NODE_GRAPH[src_node])))])
		return dst_node

	# Training step
	def step(self, action):
		temp_node_storage = copy.deepcopy(self.Node_storage)
		temp_node_state = copy.deepcopy(self.node_state)
		temp_link_state = copy.deepcopy(self.link_state)

		candid_node = []	# All virtual nodes
		candid_link = []	# All virtual links
		candid_link_dir = [] # All virtual link direction
		mig_path = []	    # All migration paths  
		mig_path_dir = []   # All migration path direction 
		mig_time = 0	    # Migration time
		E2E_delay = 0	    # Propatation delay
		self.cal_power = 0
		self.tran_power = 0
		if(self.current_SFC_index == 0):
			self.total_E2E_delay = 0
			self.total_mig_time = 0
			self.max_migration_time = 0
			self.service_time = 0
			self.Placement_Down_Time = 0
			self.Migration_Down_Time = 0
			self.total_SRT = 0
			self.p_SFC_energy = 0
			self.m_SFC_energy = 0
			self.m_mig_energy = 0
			self.deploy_node = []
			self.deploy_link = []
			self.deploy_link_dir = []
			self.vnf_need_mig_count = 0
			self.target_node = self.event.dst_node # The target node of finding the shortest path
			
		self.log_data_lst = []
		
		# time = self.event.time              # Time of deeling this event
		# user = self.event.user              # User of this event		
		# res_time = self.event.res_time      # User's residient time		
		# src_node = self.event.src_node      # User's source node		
		# dst_node = self.event.dst_node      # User's destination node		
		# SFC_type = self.event.SFC_type      # The SFC type used by user

		SFC_len = SFC_size_lst[self.event.SFC_type]   # Length of SFC
		current_SFC_index = self.current_SFC_index
		
		#placement_event_SFC or migration_event_SFC 他們的串列長度為3 每個index代表的是甚麼? #no_p
		if(current_SFC_index == 0):
			if(self.event.is_placement()):
				self.placement_event_SFC[self.event.SFC_type] += 1
			else:
				self.migration_event_SFC[self.event.SFC_type] += 1

		###############################  Adjust migration cnadid link  ####################################
		# print("try old")
		if(self.event.is_migration() and current_SFC_index==0):
			already_satisfy, E2E_delay = self.try_old_placement(temp_node_storage, temp_node_state, temp_link_state)
		else:
			already_satisfy = False

		if(self.event.is_placement() or (not already_satisfy)):

			###############################  Find the virtual nodes and virtual links  ####################################
			temp_node_storage = copy.deepcopy(self.Node_storage)
			temp_node_state = copy.deepcopy(self.node_state)
			temp_link_state = copy.deepcopy(self.link_state)
			if ( (current_SFC_index == 0) and self.event.is_migration()):
				# Release past SFC resourcee)
				past_node = copy.deepcopy(self.past_user_lst[self.event.user][0])
				past_link = copy.deepcopy(self.past_user_lst[self.event.user][1])
				past_link_dir = copy.deepcopy(self.past_user_lst[self.event.user][2])
				temp_node_storage, temp_node_state, temp_link_state = self.release_resource(self.event.user,
																				self.event.SFC_type,
																				0,
																				temp_node_storage,
																				temp_node_state,
																				temp_link_state,
																				past_node,
																				past_link,
																				past_link_dir)
				
			# Find the shortest path between user's location and all VNF			
			c_node, c_path, c_path_dir = self.find_max_prior_node(
																  temp_link_state,
																  temp_node_state,
																  temp_node_storage,
																  action,
																  self.target_node,
																  current_SFC_index)
			
			
			if c_node<NUM_SPACE_NODES:
				self.SFC_type_count[0][self.event.SFC_type] += 1
			elif c_node<NUM_SPACE_NODES+NUM_AIR_NODES:
				self.SFC_type_count[1][self.event.SFC_type] += 1
			else:
				self.SFC_type_count[2][self.event.SFC_type] += 1
	
			original_storage = copy.deepcopy(self.Node_storage[c_node])
			candid_node.append(c_node)
			candid_link.append(c_path)
			candid_link_dir.append(c_path_dir)
			self.deploy_node.append(c_node)
			self.deploy_link.append(c_path)
			self.deploy_link_dir.append(c_path_dir)
			self.target_node = c_node

			if(current_SFC_index == SFC_len - 1):
				# Find the shortest path between the last node of VNF and User's location
				c_path, c_path_dir = self.find_shortest_path(temp_link_state, c_node, self.event.dst_node, SFC_packetSize_lst[self.event.SFC_type])
				candid_link.append(c_path)
				candid_link_dir.append(c_path_dir)
				# Adjust the number of temp user in each links link_state error
				# temp_link_state = self.adjust_link_state(self.event.SFC_type, 0, temp_link_state, [c_path], [c_path_dir], 1)

			############################  Check the requriement for SFC  #################################
			past_node =  copy.deepcopy(self.past_user_lst[self.event.user][0])
			past_link = copy.deepcopy(self.past_user_lst[self.event.user][1])
			past_link_dir = copy.deepcopy(self.past_user_lst[self.event.user][2])

			# allocate node and link
			temp_node_storage, temp_node_state, temp_link_state = self.single_allocate_resource(self.event.user,
																			self.event.SFC_type,
																			0,
																			temp_node_storage,
																			temp_node_state,
																			temp_link_state,
																			candid_node,
																			candid_link,
																			candid_link_dir,
																			current_SFC_index)
			
			if(candid_node):
				for node in candid_node:
					self.Node_Utilization[node] = self.Node_Utilization[node] + 1
					if node < NUM_SPACE_NODES:
						self.node_s_used += 1
					elif node < NUM_SPACE_NODES + NUM_AIR_NODES:
						self.node_a_used += 1
					else:
						self.node_g_used += 1
					self.node_used += 1

			wireless_delay = SFC_packetSize_lst[self.event.SFC_type] / MAX_WIRELESS_BANDWIDTH
			# Count E2E_delay
			# print('e2e')
			E2E_delay = self.count_E2E_delay(temp_node_state, temp_link_state, candid_node, candid_link, candid_link_dir, self.event.SFC_type)
			# print('delay')
			if(self.current_SFC_index != 0):
				# Count SRT
				# print('SRT')
				SRT_delay = self.count_E2E_delay(temp_node_state, temp_link_state, candid_node, candid_link, candid_link_dir, self.event.SFC_type)
				# print('delay')
			if(self.current_SFC_index == SFC_len - 1):
				E2E_delay = E2E_delay + wireless_delay
				SRT_delay = SRT_delay + wireless_delay

			# reward of placement event 
			if(self.event.is_placement()):
				self.cur_mig_times_lst[self.event.user] = 0

				self.total_E2E_delay += E2E_delay
				if (current_SFC_index != 0):
					self.Placement_Down_Time += SRT_delay
				# r = (1 - (E2E_delay/1000) / res_time)    # r=original  
				# r = (1 - E2E_delay / res_time)           # r=ser_rate
				# p_reward
				v_link_conut = 1
	
				if current_SFC_index == SFC_len-1:
					v_link_conut = 2
					if self.total_E2E_delay > self.E2E_delay_req:
						self.E2E_delay_violate += 1
					if self.Placement_Down_Time > self.SD_req:
						self.PSD_violate += 1
				
				vnf_enenrgy = self.count_traverse_energy(candid_node, candid_link, candid_link_dir, self.event.SFC_type, temp_node_state, temp_link_state)
				self.p_SFC_energy += vnf_enenrgy
				# r = - vnf_enenrgy * E2E_delay
				r = (self.SD_req - E2E_delay) / vnf_enenrgy #- self.cal_avg_env_E2E(temp_node_state, temp_link_state)

	
				self.service_time = (self.event.res_time - E2E_delay)  #毫秒
				self.user_data_lst[self.event.user] =  SFC_produce_data_lst[self.event.SFC_type]
				

				
				self.placement_event_SFC_sucess[self.event.SFC_type] += 1

			# reward of migration event 2
			else:
				self.cur_mig_times_lst[self.event.user] += 1
				# Record the times of migration
				if current_SFC_index == 0:
					self.times_of_mig += 1
					if len(np.where(np.array(past_node)<NUM_SPACE_NODES)[0]) > 0:
						self.space_migrate += 1
					elif len(np.where(np.array(past_node)<NUM_SPACE_NODES+NUM_AIR_NODES)[0]) > 0:
						self.air_migrate += 1

				c_path, c_path_dir = self.find_shortest_path(temp_link_state,
															past_node[current_SFC_index],
															candid_node[0],
															SFC_lst[self.event.SFC_type][current_SFC_index])

				mig_path.append(c_path)
				mig_path_dir.append(c_path_dir)
				temp_node_storage, temp_node_state, temp_link_state = self.single_allocate_resource(self.event.user,
																				self.event.SFC_type,
																				1,
																				temp_node_storage,
																				temp_node_state,
																				temp_link_state,
																				[],
																				mig_path,
																				mig_path_dir,
																				current_SFC_index)
	
				mig_time_lst = self.count_mig_time(self.event.SFC_type, [self.user_data_lst[self.event.user]], temp_link_state, mig_path, mig_path_dir)
				mig_time = max(mig_time_lst)
				if mig_time > self.max_migration_time:
					self.max_migration_time = mig_time
				if(current_SFC_index == SFC_len - 1):
					self.Migration_Down_Time = self.max_migration_time + SRT_delay #/ 1000 #毫秒
				self.total_mig_time = self.max_migration_time

				if(mig_time > self.event.res_time):
					print("Step migration failed!")
					print(self.event)
					print(f"mig_time:{mig_time}")
					exit(1)

				# Resource(SFC & migration path) event = [time, user, res_time, dst_node, SFC type, mig_path, mig_path_dir, mig_time, residual_VNF]
				# Enqueue resource event for user
				
				for i in range(len(mig_path)):
					if(mig_path[i]):
						self.vnf_need_mig_count += 1
						event = Event.new_resource(
									user=self.event.user,
									time=self.event.time + mig_time_lst[i],
									res_time=self.event.res_time,
									dst_node=self.event.dst_node,
									SFC_type=self.event.SFC_type,
									mig_path=mig_path[i],
									mig_path_dir=mig_path_dir[i],
									mig_time=mig_time_lst[i],
									residual_VNF=self.user_data_lst[self.event.user]
								)
						self.event_queue.append(event)
				self.total_E2E_delay += E2E_delay
	
				# m_reward 2
				v_link_conut = 1
				if current_SFC_index == SFC_len-1:
					v_link_conut = 2
					if self.total_E2E_delay > self.E2E_delay_req:
						self.E2E_delay_violate += 1
					if self.Migration_Down_Time > self.SD_req:
						self.MSD_violate += 1	
					
				vnf_enenrgy = self.count_traverse_energy(candid_node, candid_link, candid_link_dir, self.event.SFC_type, temp_node_state, temp_link_state)
				mig_energy = self.count_mig_energy( mig_path, mig_path_dir, self.event.SFC_type, temp_link_state , [self.user_data_lst[self.event.user]])
				self.m_SFC_energy += vnf_enenrgy
				self.m_mig_energy += mig_energy
				r = (self.SD_req-(mig_time + E2E_delay)) /  (mig_energy +  vnf_enenrgy) # - self.cal_avg_env_E2E(temp_node_state, temp_link_state)	
				
				self.service_time = (self.event.res_time - mig_time - E2E_delay)
				self.user_data_lst[self.event.user] =  SFC_produce_data_lst[self.event.SFC_type] 

				self.migration_event_SFC_sucess[self.event.SFC_type] += 1
			# ENDIF migration(重新佈署) 
			
			# Update past_user_lst
			#更新過去的部屬的紀錄
			#這原先是一整串的candid_node，candid_link
			dcopy_candid_node = copy.deepcopy(candid_node)
			dcopy_candid_link = copy.deepcopy(candid_link)
			dcopy_candid_link_dir = copy.deepcopy(candid_link_dir)

			if(len(self.past_user_lst[self.event.user][0]) < SFC_len):
				self.past_user_lst[self.event.user][0].append(dcopy_candid_node[0])
			else:
				self.past_user_lst[self.event.user][0][current_SFC_index] = dcopy_candid_node[0]
			for temp_index in range(len(candid_link)):
				if(len(self.past_user_lst[self.event.user][1]) < SFC_len + 1):
					self.past_user_lst[self.event.user][1].append(dcopy_candid_link[temp_index])
					self.past_user_lst[self.event.user][2].append(dcopy_candid_link_dir[temp_index])
				else:
					self.past_user_lst[self.event.user][1][current_SFC_index + temp_index] = dcopy_candid_link[temp_index]
					self.past_user_lst[self.event.user][2][current_SFC_index + temp_index] = dcopy_candid_link_dir[temp_index]
			self.past_user_lst[self.event.user][3] = self.event.SFC_type
			self.past_user_lst[self.event.user][4] = self.event.dst_node
			self.past_user_lst[self.event.user][5] = E2E_delay
			#self.past_user_lst[self.event.user] = [copy.deepcopy(candid_node), copy.deepcopy(candid_link), copy.deepcopy(candid_link_dir), self.event.SFC_type, dst_node, E2E_delay]

			#################################  Loading information  ######################################
			if (current_SFC_index == SFC_len - 1):
				# storage loading
				node_loading = []
				for i in range(MAX_NODES):
					node_loading.append( self.Node_storage[i]/self.Max_Node_storage[i])

				"""if (self.event.user == 0):
				# log infomation
					self.log_data_lst.append(self.event.src_node)
					self.log_data_lst.append(self.event.dst_node)
					self.log_data_lst.append(candid_node)
					self.log_data_lst.append(candid_link)
					self.log_data_lst.append(candid_link_dir)
					self.log_data_lst.append(node_loading)
					self.log_data_lst.append(E2E_delay)
					if (self.event.is_placement()):
						self.log_data_lst.append(SRT_delay)
						self.log_data_lst.append(0)
					else:
						self.log_data_lst.append(0)
						self.log_data_lst.append(SRT_delay + mig_time)
					self.log_data_lst.append(self.event.SFC_type)"""

				#################################  Generate new event  ######################################

				# Placement event or The VNF in the migration event do not need to be migrated.
				if(self.event.is_placement() or self.vnf_need_mig_count == 0):
					# Enqueue migration event
					src_node = self.event.dst_node
					dst_node = self.random_dst_node(src_node)
					event = Event.new_migration(
       									user=self.event.user,
									   	time=self.event.time+self.event.res_time,
									   	res_time=self.event.res_time,
										src_node=src_node,
										dst_node=dst_node,
									   	SFC_type=self.event.SFC_type
                              		)
					self.event_queue.append(event)

		else: # reward of migration event 1
			temp_node_storage = copy.deepcopy(self.Node_storage)
			temp_node_state = copy.deepcopy(self.node_state)
			temp_link_state = copy.deepcopy(self.link_state)
	
			past_node = self.past_user_lst[self.event.user][0]
			past_link = self.past_user_lst[self.event.user][1]
			past_link_dir = self.past_user_lst[self.event.user][2]
			past_SFC_type = self.past_user_lst[self.event.user][3]
   
			self.cur_mig_times_lst[self.event.user] += 1
			mig_time = 0

			self.total_E2E_delay = E2E_delay
			SRT_delay = self.count_E2E_delay(temp_node_state, temp_link_state, past_node, past_link[1:], past_link_dir[1:], past_SFC_type)
			# print(f"SRT_delay={SRT_delay}, E2E_delay={E2E_delay}")
			self.Migration_Down_Time = mig_time + SRT_delay

			if self.total_E2E_delay > self.E2E_delay_req:
				self.E2E_delay_violate += 1
			if self.Migration_Down_Time > self.SD_req:
				self.MSD_violate += 1
      
			# self.Migration_Down_Time = mig_time + E2E_delay / 1000
			self.service_time = (self.event.res_time - mig_time - E2E_delay) 
			self.migration_event_SFC_sucess[self.event.SFC_type] += 1
			self.no_need_migration += 1
	
			
			vnf_enenrgy = self.count_traverse_energy(past_node, past_link, past_link_dir, past_SFC_type, temp_node_state, temp_link_state)
			mig_energy = 0
			self.m_SFC_energy += vnf_enenrgy
			self.m_mig_energy += mig_energy
   
			# r = - ( mig_energy +  vnf_enenrgy) * (mig_time + E2E_delay)
			r = (self.SD_req-(mig_time + E2E_delay)) /  (mig_energy +  vnf_enenrgy)
			# r = 0
			# r = - ( mig_energy + vnf_enenrgy  )

			
			for i in range(len(past_node)):
				if past_node[i] < NUM_SPACE_NODES:
					self.space_no_migrate += 1
					break
				elif past_node[i] < NUM_SPACE_NODES+NUM_AIR_NODES:
					self.air_no_migrate += 1
					break
					
			self.user_data_lst[self.event.user] =  SFC_produce_data_lst[self.event.SFC_type] 

		# Count max exist user
		self.count_exist_user()

		if(self.max_mig_times_lst[self.event.user] < self.cur_mig_times_lst[self.event.user]):
			self.max_mig_times_lst[self.event.user] = self.cur_mig_times_lst[self.event.user]
		if (current_SFC_index != 0):
			self.response_time = self.count_E2E()
			self.average_migration_time = self.count_MDT()
		if(current_SFC_index == SFC_len - 1):
			self.avg_env_E2E = self.cal_avg_env_E2E(temp_node_state, temp_link_state)
   
		# process resource event
		if (current_SFC_index == SFC_len - 1 or already_satisfy):
		###################################  Release migration resource  ########################################
			while(True):
				# Find next event to do
				next_event = self.event_queue[0]
				for ev in self.event_queue:
					if ev.time < next_event.time:
						next_event = ev
				self.event_queue.remove(next_event)

				# Resource(SFC & migration path) event = [time, user, res_time, dst_node, SFC type, mig_path, mig_path_dir, mig_time, residual_VNF]
				# If next event = resource event
				if next_event.is_resource():

					user = next_event.user
					SFC_type = next_event.SFC_type
					mig_path = next_event.mig_path
					mig_path_dir = next_event.mig_path_dir

					check = True

					temp_node_storage, temp_node_state, temp_link_state = self.release_resource(user, SFC_type, 1, temp_node_storage, temp_node_state, temp_link_state, [], [mig_path], [mig_path_dir])

					# temp_check = self.check_result_after_MIG_impact(temp_node_state, temp_link_state, mig_path, mig_path_dir, 0)

					# if(not temp_check):
					# 	print("check result mig error!")
					# 	exit(1)

					for ev in self.event_queue:
						if ev.is_resource() and ev.user == user:
							check = False
							break
					# if(check and np.random.random() > PLC_PROBILITY):
					if check:
						res_time = next_event.res_time
						mig_time = next_event.mig_time

						src_node = next_event.dst_node
						dst_node = self.random_dst_node(src_node)
						SFC_type = next_event.SFC_type

						# Enqueue new migration event
						event = Event.new_migration(
       									user=user,
									   	time=next_event.time + (res_time - mig_time),
									   	res_time=res_time,
										src_node=src_node,
										dst_node=dst_node,
									   	SFC_type=SFC_type
                              		)
				
						self.event_queue.append(event)

				else:
					self.event = next_event
					break

		self.Node_storage = copy.deepcopy(temp_node_storage)
		self.node_state = copy.deepcopy(temp_node_state)
		self.link_state = copy.deepcopy(temp_link_state)
     
		residual_link_state = []
		for link in range(MAX_LINK):
			for dir in range(2):
				data_rate = self.Max_link_state[link][dir] * (SFC_dr_lst[self.event.SFC_type] / (self.link_state[link][dir] + SFC_dr_lst[self.event.SFC_type])) 
				residual_link_state.append(data_rate/10**9)

		residual_CPU_state = []
		for node in range(MAX_NODES):
			data_rate = self.Max_CPU_state[node] * (SFC_dr_lst[self.event.SFC_type] / (self.node_state[node] + SFC_dr_lst[self.event.SFC_type]))
			residual_CPU_state.append(data_rate/10**9)

		if current_SFC_index == SFC_len-1 or already_satisfy:
			self.E2E_delay_req = np.random.normal(E2E_DELAY_MEAN[self.event.SFC_type])
			self.past_user_E2E_delay_req[self.event.user] = self.E2E_delay_req
			if self.event.is_placement():
				self.SD_req = self.E2E_delay_req
			else:
				self.SD_req = np.random.normal(MSD_MEAN[self.event.SFC_type])

		s_temp = residual_CPU_state + residual_link_state + [self.event.res_time, self.event.dst_node, self.event.SFC_type]  + [ self.SD_req , self.E2E_delay_req ]
		s_ = np.array([s_temp])

		self.observation_space = s_
		self.episode_complete += 1

		self.backup_s = s_
		return s_, r, already_satisfy
