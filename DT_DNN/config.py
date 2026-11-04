#import topology
import sys
import json

#TD3 parameter

MAX_EPISODES = 2000
MAX_EP_STEPS = 500
MAX_INT = sys.maxsize

METHOD = ['TD3', 'Random', 'Greedy'] #'TD3', 'Random', 'Greedy'

"""GAMMA_DT = 0.995
LR_A_DT = 0.000005
LR_C_DT = 0.0001
TAU_DT = 0.005
MEMORY_CAPACITY_DT = 30000
START_LEARNING_DT = 4000
BATCH_SIZE_DT = 192

GAMMA_DNN = 0.995
LR_A_DNN = 0.0001
LR_C_DNN = 0.0002
TAU_DNN = 0.005
MEMORY_CAPACITY_DNN = 100000
START_LEARNING_DNN = 6000
BATCH_SIZE_DNN = 128"""

GAMMA_DT = 0.9
LR_A_DT = 0.000005
LR_C_DT = 0.0001
TAU_DT = 0.01
BATCH_SIZE_DT = 128

GAMMA_DNN = 0.9
LR_A_DNN = 0.000005
LR_C_DNN = 0.0001
TAU_DNN = 0.01
BATCH_SIZE_DNN = 64

BS_STORAGE = 240000 * 8*10**3 # bits(240 GB) #基地台儲存空間

BS_COMPUTING =100 * 10**3 # cycles/s(3000 GHz) #基地台算力

BS_BANDWIDTH = 50 * 10**3 # bps(1000 Gbps) #基地台頻寬

INF = 99999999999.99999

TIME_SLOT = 1000 # ms(1秒) #時間槽長度

MAX_TIME = 80 * TIME_SLOT # ms(40秒) #最大模擬時間

#DT model
NUM_DT = 8 #環境中DT數量

MIN_UPDATE_DATA_SIZE = 400 * 8 # bits(300 MB) #DT更新資料大小的最小值
MAX_UPDATE_DATA_SIZE = 500 * 8 # bits(500 MB)  #DT更新資料大小的最大值

MIN_DT_SIZE = 40 * 8 * 10**2 # Mbits(5 GB) #DT大小的最小值
MAX_DT_SIZE = 50 * 8 * 10**2 # bits(10 GB) #DT大小的最大值

DT_PREPROC_RATE = 0.1 #DT預處理率，表示DT預處理後的資料大小占原始資料大小的比例，(用於計算DNN訓練的資料量)

MIN_DT_PROC_REQ = 650 * 8 # Mcycles(1000 Mcycles) #DT處理要求的最小值
MAX_DT_PROC_REQ = 850 * 8 # Mcycles(5000 Mcycles) #DT處理要求的最大值

MIN_DT_DR_REQ = 2 * 10**6 # Mbits/s
MAX_DT_DR_REQ = 3 * 10**6 # Mbits/s

IOT_STATIC = 0

IOT_MOVE_PROB = 0.9 # IoT移動的概率，表示每個時間槽中，每個IoT節點有50%的概率會移動到另一個節點

# User model
MAX_USER = [20, 40, 60, 80] #最大使用者數量
MIN_DELAY_CONSTRAINT = 3000 # ms
MAX_DELAY_CONSTRAINT = 1000 # ms

MIN_DR_REQ = 4 * 10**6 # bits/s
MAX_DR_REQ = 5 * 10**6 # bits/s

MIN_PROC_REQ = 1600 * 8 # Mcycles(500 Mcycles)
MAX_PROC_REQ = 1800 * 8 # Mcycles(2000 Mcycles)

MIN_ACCURACY_REQUIREMENT = 0.7 
MAX_ACCURACY_REQUIREMENT = 0.9

#DNN model
#NUM_DNN = 3 #環境中DNN數量
DNN_SIZE = 20 * 8 * 10**3 # Mbits(20 GB)
DNN_RESULT_SIZE_RATE = 0.8 #DNN結果大小占DNN訓練資料大小的比例

MAX_ACC = 0.95
MIN_ACC = 0.45

BETA_ACC = 1

# Event model
EVENT_DENSITY = 8000  #ms(第80秒)

# Netwrok model
MAX_NODES = 18 #最大節點數
MAX_WIRELESS_BANDWIDTH = 50 # Mbps(50 Mbps) #無線頻寬

MEMORY_CAPACITY_DT = MAX_TIME / TIME_SLOT * 8 * 50

START_LEARNING_DT = MAX_TIME / TIME_SLOT * 8 * 50

MEMORY_CAPACITY_DNN = MAX_TIME / EVENT_DENSITY * 50

START_LEARNING_DNN = MAX_TIME / EVENT_DENSITY * 100

MEC_Node_Graph_lst = []
MEC_Link_Graph_lst = []
Link_propagation_lst = []
link_num_lst = []

def read_log_file(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

path = "topology/"

file_name = ["Node_Graph", "Link_Graph", "Propagation_Delay", "Link_Num"]

MEC_Node_Graph_lst = read_log_file(path + file_name[0] + ".txt")
MEC_Link_Graph_lst = read_log_file(path + file_name[1] + ".txt")
Link_propagation_lst = read_log_file(path + file_name[2] + ".txt")
link_num_lst = read_log_file(path + file_name[3] + ".txt")

MEC_NODE_GRAPH = MEC_Node_Graph_lst[0]
MEC_LINK_GRAPH = MEC_Link_Graph_lst[0]
LINK_PROPAGATION = Link_propagation_lst[0]
MAX_LINK = link_num_lst[0]

def get_graph_info(episode):
    global MEC_NODE_GRAPH, MEC_LINK_GRAPH, LINK_PROPAGATION, MAX_LINK
    MEC_NODE_GRAPH = MEC_Node_Graph_lst[episode]
    MEC_LINK_GRAPH = MEC_Link_Graph_lst[episode]
    LINK_PROPAGATION = Link_propagation_lst[episode]
    MAX_LINK = link_num_lst[episode]

