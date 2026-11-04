#import tensorflow as tf
from cProfile import label
import tensorflow.compat.v1 as tf
import numpy as np
import random
import math
import os
import matplotlib.pyplot as plt
# from env_SFC_no_algorithm import SFC
from env_SAGIN import ENV as SFC
import sys
# from DDPG_topology import path
import pickle
import time
import datetime
import gc
UE = 250
description_str = ""
if len(sys.argv) > 1:
    description = sys.argv[1:]
    for i in description:
        if 'ue' in i:
            UE = int(i[2:])
        description_str += i + '_'
if UE == 0:
    UE = int(input('請輸入UE數量:'))

MAX_USER = UE

AGENT_NUM = 3

FED_AVAILABLE = True

TIMES_FOR_AVERAGE = 30
    
# path = "./result/DDPG/" + datetime.datetime.now().strftime("%m-%d_%H-%M-%S_") + description_str + "/"
path = "./result/DDPG/" + description_str + "/"

start = time.time()


#測試是哪個gpu or not
gpu_device_name = tf.test.gpu_device_name()
print(gpu_device_name)
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

tf.disable_eager_execution()
np.random.seed(1)
tf.set_random_seed(1)

GLOBAL_SESS = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))

#####################  hyper parameters  ####################
env2 = SFC(UE)
save_dir = path #"result/first_compare/no_algorithm/DDPG/"+str(env2.MAX_USER)+"_User/"
MAX_EPISODES = 800 #不用訓練到80000，可以訓練3000
MAX_EP_STEPS = UE * 6
# MAX_EP_STEPS = env2.MAX_USER*7.7 #for 服務連續性，不同數量的使用者 可以跑相同的比例的migration(服務連續)。
# MAX_EP_STEPS = int(MAX_EP_STEPS)
#如果step 與user有關係
#服務連續性指的是使用者migration中斷率 ex 只有一個user 走10步 都不會斷 100%不中斷
#                     #ex 只有一個user 走10步 有一步不會斷 90%不中斷 10%中斷

TIMES_OF_DO_AVERAGE = 10
TIMES_OF_DO_OUTPUT = 100

TIMES_OF_DO_SFC_AVERAGE = 100

GAMMA = 0.9     # reward discount
OUTPUT_GRAPH = False

MAX_INT = sys.maxsize

#DDPG
DDPG_LR_A = 0.00000000005    # learning rate for actor
DDPG_LR_C = 0.0000000001    # learning rate for critic

# DDPG_LR_A = 0.00000000000005    # learning rate for actor
# DDPG_LR_C = 0.0000000000001    # learning rate for critic

TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 100000
BATCH_SIZE = 32

###############  Record average reward graph data  ###############
    
###############################  DDPG  ####################################
                    
###############################  DDPG  ####################################
                    
class DDPG2_placement(object):
    def __init__(self, a_dim, s_dim, a_bound, agent_id, sess):
        self.sess = sess # 新增 agent_id 參數
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()

        self.agent_id = agent_id

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        
        # 為每個 Agent 建立獨立的命名空間
        """with tf.variable_scope(f'Agent_P_{self.agent_id}'):
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')

            with tf.variable_scope('Actor1_t'):
                self.a = self._build_a(self.S, scope='eval1', trainable=True)
                a_ = self._build_a(self.S_, scope='target1', trainable=False)
            with tf.variable_scope('Critic1_t'):
                q = self._build_c(self.S, self.a, scope='eval1', trainable=True)
                q_ = self._build_c(self.S_, a_, scope='target1', trainable=False)

        # 更新參數獲取範圍，確保只抓取該 Agent 內部的變數
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_P_{self.agent_id}/Actor1_t/eval1')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_P_{self.agent_id}/Actor1_t/target1')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_P_{self.agent_id}/Critic1_t/eval1')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_P_{self.agent_id}/Critic1_t/target1')"""

        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope(f"Agent{self.agent_id}/Actorp/eval"):
            self.a = self._build_a(self.S, scope='eval', trainable=True)

        with tf.variable_scope(f"Agent{self.agent_id}/Actorp/target"):
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope(f"Agent{self.agent_id}/Criticp/eval"):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)

        with tf.variable_scope(f"Agent{self.agent_id}/Criticp/target"):
            q_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)
        # networks parameters
        self.ae_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Actorp/eval"
            )
        self.at_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Actorp/target"
            )
        self.ce_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Criticp/eval"
            )
        self.ct_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Criticp/target"
            )

        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(DDPG_LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(DDPG_LR_A).minimize(a_loss, var_list=self.ae_params)

        """self.sess.run(tf.global_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 讓 TF 只佔用需要的顯存，而不是一次全佔
        self.sess = tf.Session(config=config)"""

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})

    def learn(self):
        self.sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s[0].tolist(), a[0].tolist(), [r], s_[0].tolist()))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 600, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            a = tf.layers.dense(l5, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 600
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            q = tf.layers.dense(l5, 1, trainable=trainable)
            return q

class DDPG2_migration(object):
    def __init__(self, a_dim, s_dim, a_bound, agent_id, sess):
        self.sess = sess # 新增 agent_id 參數
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        #self.sess = tf.Session()
        self.agent_id = agent_id

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        
        """with tf.variable_scope(f'Agent_M_{ self.agent_id}'): # 獨立作用域
            self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
            self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
            self.R = tf.placeholder(tf.float32, [None, 1], 'r')

            with tf.variable_scope('Actor2_t'):
                self.a = self._build_a(self.S, scope='eval2', trainable=True)
                a_ = self._build_a(self.S_, scope='target2', trainable=False)
            with tf.variable_scope('Critic2_t'):
                q = self._build_c(self.S, self.a, scope='eval2', trainable=True)
                q_ = self._build_c(self.S_, a_, scope='target2', trainable=False)

        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_M_{ self.agent_id}/Actor2_t/eval2')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_M_{ self.agent_id}/Actor2_t/target2')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_M_{ self.agent_id}/Critic2_t/eval2')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=f'Agent_M_{ self.agent_id}/Critic2_t/target2')"""

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope(f"Agent{self.agent_id}/Actorm/eval"):
            self.a = self._build_a(self.S, scope='eval', trainable=True)

        with tf.variable_scope(f"Agent{self.agent_id}/Actorm/target"):
            self.a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope(f"Agent{self.agent_id}/Criticm/eval"):
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)

        with tf.variable_scope(f"Agent{self.agent_id}/Criticm/target"):
            q_ = self._build_c(self.S_, self.a_, scope='target', trainable=False)

        # networks parameters
        self.ae_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Actorm/eval"
            )
        self.at_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Actorm/target"
            )
        self.ce_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Criticm/eval"
            )
        self.ct_params = tf.get_collection(
                tf.GraphKeys.TRAINABLE_VARIABLES,
                scope=f"Agent{self.agent_id}/Criticm/target"
            )

        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(DDPG_LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)
        self.atrain = tf.train.AdamOptimizer(DDPG_LR_A).minimize(a_loss, var_list=self.ae_params)

        """self.sess.run(tf.global_variables_initializer())
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True  # 讓 TF 只佔用需要的顯存，而不是一次全佔
        self.sess = tf.Session(config=config)
"""
    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s})

    def learn(self):
        self.sess.run(self.soft_replace)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s[0].tolist(), a[0].tolist(), [r], s_[0].tolist()))
        index = self.pointer % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            l1 = tf.layers.dense(s, 600, activation=tf.nn.relu, name='l1', trainable=trainable)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            a = tf.layers.dense(l5, self.a_dim, activation=tf.nn.sigmoid, name='a', trainable=trainable)
            return a

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 600
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            l1 = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            l2 = tf.layers.dense(l1, 300, activation=tf.nn.relu, name='l2', trainable=trainable)
            l3 = tf.layers.dense(l2, 200, activation=tf.nn.relu, name='l3', trainable=trainable)
            l4 = tf.layers.dense(l3, 100, activation=tf.nn.relu, name='l4', trainable=trainable)
            l5 = tf.layers.dense(l4, 50, activation=tf.nn.relu, name='l5', trainable=trainable)
            q = tf.layers.dense(l5, 1, trainable=trainable)
            return q


###############################  Federated helper  ####################################

def get_actor_weights(agent):
    """
    Return a list of numpy arrays representing the actor (eval) weights
    for a TF1 DDPG agent (placement or migration).
    """
    return agent.sess.run(agent.ae_params)


def get_critic_weights(agent):
    """Return the critic (eval) weights for a TF1 DDPG agent."""
    return agent.sess.run(agent.ce_params)


def _ensure_assign_handles(agent, cache_attr, param_vars):
    handles = getattr(agent, cache_attr, None)
    if handles is None:
        placeholders = []
        ops = []
        for var in param_vars:
            ph = tf.placeholder(dtype=var.dtype, shape=var.shape)
            placeholders.append(ph)
            ops.append(tf.assign(var, ph))
        handles = (placeholders, ops)
        setattr(agent, cache_attr, handles)
    return handles


def _assign_params(agent, param_vars, new_params, cache_attr):
    placeholders, ops = _ensure_assign_handles(agent, cache_attr, param_vars)
    feed_dict = {ph: value for ph, value in zip(placeholders, new_params)}
    agent.sess.run(ops, feed_dict=feed_dict)


def _sync_target_network(agent):
    sync_ops = getattr(agent, "_target_sync_ops", None)
    if sync_ops is None:
        sync_ops = []
        for t, e in zip(agent.at_params, agent.ae_params):
            sync_ops.append(tf.assign(t, e))
        for t, e in zip(agent.ct_params, agent.ce_params):
            sync_ops.append(tf.assign(t, e))
        setattr(agent, "_target_sync_ops", sync_ops)
    agent.sess.run(sync_ops)


def assign_actor_weights(agent, new_params, sync_target=True):
    """
    Replace the actor (eval) weights of a TF1 DDPG agent with `new_params`.
    Set `sync_target=False` if you want to rely on the usual soft update instead.
    """
    _assign_params(agent, agent.ae_params, new_params, "_actor_assign_handles")
    if sync_target:
        _sync_target_network(agent)


def assign_critic_weights(agent, new_params, sync_target=True):
    """Replace critic weights and optionally sync the target critic."""
    _assign_params(agent, agent.ce_params, new_params, "_critic_assign_handles")
    if sync_target:
        _sync_target_network(agent)


def federated_average(param_sets, client_weights=None):
    """
    FedAvg over a list of parameter lists returned by `get_actor_weights`/`get_critic_weights`.
    """
    if not param_sets:
        raise ValueError("param_sets must contain at least one client.")
    num_clients = len(param_sets)
    if client_weights is None:
        normalized = [1.0 / num_clients for _ in range(num_clients)]
    else:
        if len(client_weights) != num_clients:
            raise ValueError("client_weights length must match number of clients.")
        total = float(sum(client_weights))
        if total == 0.0:
            raise ValueError("Sum of client_weights must be > 0.")
        normalized = [w / total for w in client_weights]
    averaged_params = []
    for params_tuple in zip(*param_sets):
        avg_param = np.zeros_like(params_tuple[0])
        for weight, param in zip(normalized, params_tuple):
            avg_param += weight * param
        averaged_params.append(avg_param)
    return averaged_params


def run_fedavg_and_broadcast(agent_list, client_weights=None, sync_target=True):
    """
    Collect actor/critic weights from each agent in `agent_list`, run FedAvg,
    and assign the averaged weights back to all agents.
    Returns (avg_actor_params, avg_critic_params).
    """
    if not agent_list:
        raise ValueError("agent_list must contain at least one DDPG agent.")
    actor_sets = [get_actor_weights(agent) for agent in agent_list]
    critic_sets = [get_critic_weights(agent) for agent in agent_list]
    avg_actor = federated_average(actor_sets, client_weights)
    avg_critic = federated_average(critic_sets, client_weights)
    for agent in agent_list:
        assign_actor_weights(agent, avg_actor, sync_target=sync_target)
        assign_critic_weights(agent, avg_critic, sync_target=sync_target)
        if not sync_target:
            # still softly align target nets for stability
            agent.sess.run(agent.soft_replace)
    return avg_actor, avg_critic



def check_nan(*tensors, tag=""):
    for i, t in enumerate(tensors):
        # 支援 numpy arrays or python scalars
        try:
            arr = np.array(t)
            if np.isnan(arr).any() or np.isinf(arr).any():
                print(f"NaN/Inf detected in {tag} tensor#{i}: shape={arr.shape}, sample={arr.flatten()[:10]}")
                return True
        except Exception as e:
            print("check_nan error:", e)
    return False

###############################  training  ####################################
config = tf.ConfigProto()
config.gpu_options.allow_growth = True  # 動態顯存
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 限制總顯存上限
GLOBAL_SESS = tf.Session(config=config)

env2 = []
p_DDPG = []
m_DDPG = []
for agent_id in range(AGENT_NUM):
    env2.append(SFC(MAX_USER))

    # placement parameter
    p_s_dim = env2[agent_id].observation_space.shape[1]
    p_a_dim = env2[agent_id].a_dim
    p_a_bound = np.array([[0, 1]] * env2[agent_id].a_dim)

    # migration paremeter
    m_s_dim = env2[agent_id].observation_space.shape[1]
    m_a_dim = env2[agent_id].a_dim
    m_a_bound = np.array([[0, 1]] * env2[agent_id].a_dim)

    # 初始化時傳入 agent_id
    p_DDPG.append(DDPG2_placement(p_a_dim, p_s_dim, p_a_bound, agent_id, sess=GLOBAL_SESS)) 
    m_DDPG.append(DDPG2_migration(m_a_dim, m_s_dim, m_a_bound, agent_id, sess=GLOBAL_SESS))

GLOBAL_SESS.run(tf.global_variables_initializer())
# 蒐集輸出資料----------------------------------------------------------------

#log_data_lst = []

p_r_lst_DDPG = []
m_r_lst_DDPG = []

traversal_delay_lst = []
migration_service_downtime_lst = []
placement_service_downtime_lst = []
p_episode_E2E_delay_lst = []
p_service_time_lst = []

m_episode_E2E_delay_lst = []
m_episode_mig_time_lst = []
m_service_time_lst = []

traversal_E2E_delay_lst = []

placement_energy_lst = []
migration_energy_lst = []
traversal_energy_lst = []
m_sfc_energy_lst = []

p_invalid_rate_lst = []
m_invalid_rate_lst = []
trvl_invalid_rate_lst = []

PSD_violate_lst = []
MSD_violate_lst = []
TD_violate_lst = []

nodes_users_lst = []
links_users_lst = []
node_g_used_lst = []
node_a_used_lst = []
node_s_used_lst = []
node_used_lst   = [] 
#----------------------------------------------------------------
localtime = time.localtime()
print(time.strftime("%Y-%m-%d %H:%M:%S", localtime))

if not os.path.isdir('trainData'):
    os.mkdir('trainData')
    

datadirectory = 'trainData/' + time.strftime("%m%d_%H%M%S", localtime)
if not os.path.isdir(datadirectory):
    os.mkdir(datadirectory)
    print("create directory:", datadirectory)

directories = f'{datadirectory}/DDPG_{str(MAX_USER)}_log'
if not os.path.isdir(directories):
    os.mkdir(directories)

print("environment user:", str(MAX_USER))
print("MAX_EPISODES:", str(MAX_EPISODES))
noise_p = 0.2
noise_m = 0.2
for i in range(MAX_EPISODES):
    
    start_time = time.time()

    p_episode_reward = 0
    m_episode_reward = 0

    #m_episode_mig_time = 0

    p_event_num = 0
    m_event_num = 0

    p_event_vnf_num = 0
    m_event_vnf_num = 0

    # Invalid Rate-----------------------
    p_invalid_rate = 0
    m_invalid_rate = 0
    trvl_invalid_rate = 0
    #-----------------------------------

    # Delay-------------------------------
    #average_placement_down_time = 0
    Placement_DownTime = 0
    p_episode_E2E_delay = 0
    p_service_time = 0
    average_migration_down_time = 0
    Migration_DownTime = 0
    average_traversal_delay = 0
    m_episode_E2E_delay = 0
    m_service_time = 0
    m_episode_mig_time = 0
    p_violate = 0
    m_violate = 0
    trvl_violate = 0
    #-----------------------------------

    #placement
    average_p_energy = 0
    average_m_mig_energy = 0
    p_energy = 0
    m_SFC_energy = 0
    m_mig_energy = 0

    p_service_time = 0
    m_service_time = 0

    s = []
    placement_gradient_lst = []
    placement_actor_gradient_lst = []
    placement_critic_gradient_lst = []
    migration_gradient_lst = []
    migration_actor_gradient_lst = []
    migration_critic_gradient_lst = []

    learning_p = False
    learning_m = False

    for a in range(AGENT_NUM):
        s.append(env2[a].reset())

    for j in range(MAX_EP_STEPS):
        do_average_p = FED_AVAILABLE
        do_average_m = FED_AVAILABLE
        #if math.isnan(s[0][0]):
            #os._exit()
        # Add exploration noise
        # placement
        for agent_id in range(AGENT_NUM):
            if(env2[agent_id].event.is_placement()):
                for k in range(env2[agent_id].get_SFC_len()):
                    p_event_vnf_num += 1
                    """if j % 100 == 0:
                        params = p_DDPG[agent_id].sess.run(p_DDPG[agent_id].ae_params)
                        for i, v in enumerate(params):
                            if np.isnan(v).any():
                                print(f"ACTOR PARAM #{i} BECAME NAN")"""
                    env2[agent_id].current_SFC_index = k
                    #print("Placement Step:", j, " Agent:", agent_id, " SFC index:", k)
                    if np.isnan(s[agent_id]).any() or np.isinf(s[agent_id]).any():
                        print("STATE IS NAN:", s[agent_id])

                    if check_nan(s, tag="state BEFORE action"): 
                        os._exit(1)

                    a = p_DDPG[agent_id].choose_action(s[agent_id])
                    #print(len(a))
                    #print(a)

                    if check_nan(a, tag="action AFTER choose_action"): 
                        os._exit(1)
                    a = np.clip(np.random.normal(a, noise_p), 1e-2,
                            1 - 1e-2)
                    #a = np.squeeze(a)
                    s_,r,already_satisfy = env2[agent_id].step(a)
                    # update p_DDPG
                    # print("training placement DDPG")
                    p_DDPG[agent_id].store_transition(s[agent_id], a, r * 10, s_)
                    if p_DDPG[agent_id].pointer > MEMORY_CAPACITY:
                        learning_p = True
                        noise_p *= .9995  # decay the action randomness
                        p_DDPG[agent_id].learn()

                    s[agent_id] = s_
                    p_episode_reward += r
                    
                    if(already_satisfy == True):
                        s[agent_id] = s_
                        break


                    
                p_event_num += 1

                Placement_DownTime += env2[agent_id].Placement_Down_Time

                p_energy += env2[agent_id].p_SFC_energy

                p_episode_E2E_delay += env2[agent_id].total_E2E_delay

                p_service_time += env2[agent_id].service_time

                #p_violate += env2[agent_id].PSD_violate

                ##########   Used for output   ########## 
                
            # migration
            else:
                for k in range(env2[agent_id].get_SFC_len()):
                    m_event_vnf_num += 1
                    """if j % 100 == 0:
                        params = m_DDPG[agent_id].sess.run(m_DDPG[agent_id].ae_params)
                        for i, v in enumerate(params):
                            if np.isnan(v).any():
                                print(f"ACTOR PARAM #{i} BECAME NAN")"""
                    env2[agent_id].current_SFC_index = k

                    if check_nan(s, tag="state BEFORE action"): 
                        os._exit(1)

                    a = m_DDPG[agent_id].choose_action(s[agent_id])
                    #print(len(a))
                    #a = np.squeeze(a)

                    if check_nan(a, tag="action AFTER choose_action"): 
                        os._exit(1)

                    s_,r,already_satisfy = env2[agent_id].step(a)
                    # update m_DDPG
                    if(already_satisfy == True):
                        s[agent_id] = s_
                        m_episode_reward += r
                        break
                    else:
                        # print("training migration DDPG")
                        m_DDPG[agent_id].store_transition(s[agent_id], a, r * 10, s_)
                        s[agent_id] = s_
                        m_episode_reward += r
                    if m_DDPG[agent_id].pointer > MEMORY_CAPACITY:
                        learning_m = True
                        noise_m *= .9995  # decay the action randomness
                        m_DDPG[agent_id].learn()  
    
                #if(already_satisfy != True):
                m_event_num += 1

                Migration_DownTime += env2[agent_id].Migration_Down_Time

                m_episode_E2E_delay += env2[agent_id].total_E2E_delay 

                m_episode_mig_time += env2[agent_id].total_mig_time

                m_service_time += env2[agent_id].service_time

                m_SFC_energy += env2[agent_id].m_SFC_energy

                m_mig_energy += env2[agent_id].m_mig_energy

                #m_violate += env2[agent_id].MSD_violate
                
            #env2[agent_id].reset()
            
                
            #if (i + 1) * MAX_EP_STEPS >= MEMORY_CAPACITY_m:
            if j % TIMES_FOR_AVERAGE == TIMES_FOR_AVERAGE - 1:
                if p_DDPG[agent_id].pointer <= MEMORY_CAPACITY:
                    do_average_p = False 

                if m_DDPG[agent_id].pointer <= MEMORY_CAPACITY:
                    do_average_m = False

                if agent_id == AGENT_NUM - 1:
                    if do_average_p:
                        run_fedavg_and_broadcast(p_DDPG, client_weights=None, sync_target=True)
                    if do_average_m:
                        run_fedavg_and_broadcast(m_DDPG, client_weights=None, sync_target=True)
            """if j % TIMES_FOR_AVERAGE == TIMES_FOR_AVERAGE - 1:
                        
                #placement_gradient_lst.append([get_actor_weights(p_DDPG[agent_id])], [get_critic_weights(p_DDPG[agent_id])])
                placement_actor_gradient_lst.append(get_actor_weights(p_DDPG[agent_id]))
                placement_actor_gradient_lst.append(get_critic_weights(p_DDPG[agent_id]))
                #migration_gradient_lst.append([get_actor_weights(m_DDPG[agent_id])], [get_critic_weights(m_DDPG[agent_id])])
                migration_actor_gradient_lst.append(get_actor_weights(m_DDPG[agent_id]))
                migration_actor_gradient_lst.append(get_critic_weights(m_DDPG[agent_id]))

                if agent_id == 0:
                    do_average_p = True
                    do_average_m = True
                else:
                    if len(placement_actor_gradient_lst[agent_id]) != len(placement_actor_gradient_lst[agent_id - 1]):
                        do_average_p = False
                    if len(placement_actor_gradient_lst[agent_id]) != len(placement_actor_gradient_lst[agent_id - 1]):
                        do_average_p = False
                    if len(migration_actor_gradient_lst[agent_id]) != len(migration_actor_gradient_lst[agent_id - 1]):
                        do_average_m = False
                    if len(migration_actor_gradient_lst[agent_id]) != len(migration_actor_gradient_lst[agent_id - 1]):
                        do_average_m = False

                if p_DDPG[agent_id].pointer <= MEMORY_CAPACITY:
                    do_average_p = False 

                if m_DDPG[agent_id].pointer <= MEMORY_CAPACITY:
                    do_average_m = False

                if agent_id == AGENT_NUM - 1:
                    if do_average_p:
                        placement_actor_average_gradient = federated_average(placement_actor_gradient_lst, client_weights=None)
                        placement_critic_average_gradient = federated_average(placement_critic_gradient_lst, client_weights=None)
                        for x in range(AGENT_NUM):
                            DDPG.load_ddpg_gradient_weights(p_DDPG[x], placement_average_gradient)
                    if do_average_m:
                        migration_actor_average_gradient = federated_average(migration_actor_gradient_lst, client_weights=None)
                        migration_critic_average_gradient = federated_average(migration_critic_gradient_lst, client_weights=None)
                        for x in range(AGENT_NUM):
                            DDPG.load_ddpg_gradient_weights(m_DDPG[x], migration_average_gradient)
                    #print(len(migration_gradient_lst[0]), len(migration_gradient_lst[1]), len(migration_gradient_lst[2]))

            if j % TIMES_FOR_AVERAGE == 0 and j != 0:
                DDPG.load_ddpg_gradient_weights(p_DDPG[agent_id], placement_average_gradient)
                DDPG.load_ddpg_gradient_weights(m_DDPG[agent_id], migration_average_gradient)"""

        
        
        if j == MAX_EP_STEPS - 1:
            for agent_id in range(AGENT_NUM):
                trvl_violate = env2[agent_id].E2E_delay_violate
                p_violate = env2[agent_id].PSD_violate
                m_violate = env2[agent_id].MSD_violate

            
            # Reward list
            average_p_episode_reward = p_episode_reward / p_event_num
            average_m_episode_reward = m_episode_reward / m_event_num
            p_r_lst_DDPG.append(average_p_episode_reward)
            m_r_lst_DDPG.append(average_m_episode_reward)

            # 計算Invalid Rate --------------------------------
            # Placement

            # -----------------------------------------------

            # 計算 Delay --------------------------------
            # PSD
            if (p_event_num != 0):
                average_placement_down_time = Placement_DownTime / p_event_num

                average_p_energy = p_energy / p_event_num

                average_p_episode_E2E_delay = p_episode_E2E_delay / p_event_num

                average_p_service_time = p_service_time / p_event_num

                average_p_violate = p_violate / p_event_vnf_num
            
            placement_service_downtime_lst.append(average_placement_down_time)
                
            placement_energy_lst.append(average_p_energy)

            p_episode_E2E_delay_lst.append(average_p_episode_E2E_delay)

            p_service_time_lst.append(average_p_service_time)

            PSD_violate_lst.append(average_p_violate)


            # MSD
            if (m_event_num != 0):
                average_migration_down_time = Migration_DownTime / m_event_num

                average_m_mig_energy = m_mig_energy / m_event_num

                average_m_sfc_energy = m_SFC_energy / m_event_num

                average_m_episode_E2E_delay = m_episode_E2E_delay / m_event_num

                #m_episode_mig_time 

                average_m_service_time  = m_service_time / m_event_num

                average_m_violate = m_violate / m_event_vnf_num


            migration_service_downtime_lst.append(average_migration_down_time)

            migration_energy_lst.append(average_m_mig_energy)

            m_sfc_energy_lst.append(average_m_sfc_energy)

            m_episode_E2E_delay_lst.append(average_m_episode_E2E_delay)

            m_service_time_lst.append(average_m_service_time)

            MSD_violate_lst.append(average_m_violate)

            if(m_event_num + p_event_num != 0):
                trvl_violate /= (m_event_vnf_num + p_event_vnf_num)
                trvl_average_energy = (m_SFC_energy + p_energy) / (m_event_num + p_event_num)
                trvl_average_delay = (m_episode_E2E_delay + p_episode_E2E_delay) / (m_event_num + p_event_num)

            TD_violate_lst.append(trvl_violate)

            traversal_energy_lst.append(trvl_average_energy)

            traversal_E2E_delay_lst.append(trvl_average_delay)

            nodes_users_lst.append(env2[agent_id].Node_Utilization)
            links_users_lst.append(env2[agent_id].Link_Utilization)
            node_g_used_lst.append(env2[agent_id].node_g_used)
            node_a_used_lst.append(env2[agent_id].node_a_used)
            node_s_used_lst.append(env2[agent_id].node_s_used)
            node_used_lst.append(env2[agent_id].node_used)

            print()
            print("Episode = ",i)
            print("----------------------------------------------------")
            print("Placement Episode reward = ", average_p_episode_reward)
            print("Migration Episode reward = ", average_m_episode_reward)
            print("----------------------------------------------------")
            #print("Max exist user = ",env2[agent_id].max_exist_user)
            print("----------------------------------------------------")
            print("Average traversal delay (ms) = ", average_traversal_delay)
            print("Average migration time (ms) = ", average_migration_down_time)
            print("Average placement down time (ms) = ", average_placement_down_time)
            """ # Test -------------------------------------------------------------
            print("Average migration service downtime (ms) = ", average_MSD)
            print("Average traversal delay (ms) = ", average_TD)
            #-------------------------------------------------------------- """
            print("----------------------------------------------------")
            print("----------------------------------------------------")
            """print("Placement VNF delay invalid rate = ", p_invalid_rate * 100,"%")
            print("Migration VNF delay invalid rate = ", m_invalid_rate * 100,"%")
            print("Trversal VNF delay invalid rate = ", trvl_invalid_rate * 100,"%")"""
            print("Placement Energy Consumption (J) = ", average_p_energy)
            print("Migration Energy Consumption (J) = ", average_m_mig_energy)
            print("----------------------------------------------------")
            print("Episode time: ", time.time() - start_time, "sec")
            print(f"Len of event_queue: {len(env2[0].event_queue)}, Len of past_records: {len(env2[0].past_user_lst)}")
            print("start learning p:",learning_p)
            print("start learning m:",learning_m)
            

            # 輸出log--------------------------------------------------------------
            if i % 50 == 49:
				# Log--------------------------------------------------------------
                directories = f'{datadirectory}/DDPG_{str(MAX_USER)}_log'
                if not os.path.isdir(directories):
                    os.mkdir(directories)

                """path = f'{directories}/DDPG_P_Reward.txt'
                f = open(path, 'w')
                for i in range(len(p_r_lst_DDPG)):
                    f.write(str(p_r_lst_DDPG[i]))
                    if i != len(p_r_lst_DDPG) - 1:
                        f.write(',')
                f.close()

                path = f'{directories}/DDPG_M_Reward.txt'
                f = open(path, 'w')
                for i in range(len(m_r_lst_DDPG)):
                    f.write(str(m_r_lst_DDPG[i]))
                    if i != len(m_r_lst_DDPG) - 1:
                        f.write(',')
                f.close()

                 path = f'{directories}/DDPG_Traversal_Delay_{str(MAX_USER)}.txt'
                f = open(path, 'w')
                for i in range(len(traversal_delay_lst)):
                    f.write(str(traversal_delay_lst[i]))
                    if i != len(traversal_delay_lst) - 1:
                        f.write(',')
                f.close()"""

                path = f'{directories}/DDPG_Migration_Service_Downtime.txt'
                f = open(path, 'w')
                for i in range(len(migration_service_downtime_lst)):
                    f.write(str(migration_service_downtime_lst[i]))
                    if i != len(migration_service_downtime_lst) - 1:
                        f.write(',')
                f.close()

                """path = f'{directories}/m_Average_service_time_graph_{str(MAX_USER)}.txt'
                f = open(path, 'w')

                for i in range(len(m_service_time_lst)):
                    f.write(str(m_service_time_lst[i]))
                    if i != len(m_service_time_lst) - 1:
                        f.write(',')

                f.close()"""

                path = f'{directories}/DDPG_Placement_Service_Downtime.txt'
                f = open(path, 'w')
                for i in range(len(placement_service_downtime_lst)):
                    f.write(str(placement_service_downtime_lst[i]))
                    if i != len(placement_service_downtime_lst) - 1:
                        f.write(',')
                f.close()

                """path = f'{directories}/Placement_Energy_Consumption.txt'
                f = open(path, 'w')
                for i in range(len(placement_energy_lst)):
                    f.write(str(placement_energy_lst[i]))
                    if i != len(placement_energy_lst) - 1:
                        f.write(',')
                f.close()"""

                path = f'{directories}/Migration_Energy_Consumption.txt'
                f = open(path, 'w')
                for i in range(len(migration_energy_lst)):
                    f.write(str(migration_energy_lst[i]))
                    if i != len(migration_energy_lst) - 1:
                        f.write(',')
                f.close()

                """path = f'{directories}/Average_traverse_energy_consumption_{str(MAX_USER)}.txt'
                f = open(path, 'w')
                for i in range(len((np.array(placement_energy_lst) + np.array(m_sfc_energy_lst)) / 2)):
                    f.write(str(placement_service_downtime_lst[i]))
                    if i != len(placement_service_downtime_lst) - 1:
                        f.write(',')
                f.close()"""

                path = f'{directories}/Average_Traversal_Energy_Consumption.txt'
                f = open(path, 'w')
                for i in range(len(traversal_energy_lst)):
                    f.write(str(traversal_energy_lst[i]))
                    if i != len(traversal_energy_lst) - 1:
                        f.write(',')
                f.close()

                path = f'{directories}/Placement_Violate_Rate.txt'
                f = open(path, 'w')
                for i in range(len(PSD_violate_lst)):
                    f.write(str(PSD_violate_lst[i]))
                    if i != len(PSD_violate_lst) - 1:
                        f.write(',')
                f.close()

                path = f'{directories}/Migration_Violate_Rate.txt'
                f = open(path, 'w')
                for i in range(len(MSD_violate_lst)):
                    f.write(str(MSD_violate_lst[i]))
                    if i != len(MSD_violate_lst) - 1:
                        f.write(',')
                f.close()

                path = f'{directories}/Traversal_Violate_Rate.txt'
                f = open(path, 'w')
                for i in range(len(TD_violate_lst)):
                    f.write(str(TD_violate_lst[i]))
                    if i != len(TD_violate_lst) - 1:
                        f.write(',')
                f.close()

                """path = f'{directories}/Nodes_Users.txt'
                f = open(path, 'w')
                for i in range(len(nodes_users_lst)):
                    f.write(str(nodes_users_lst[i]))
                    if i != len(nodes_users_lst) - 1:
                        f.write(',')
                f.close()

                path = f'{directories}/Links_Users_{str(MAX_USER)}.txt'
                f = open(path, 'w')
                for i in range(len(links_users_lst)):
                    f.write(str(links_users_lst[i]))
                    if i != len(links_users_lst) - 1:
                        f.write(',')
                f.close()

                # 存 ground 使用量
                path = f"{directories}/node_g_used.txt"
                f = open(path, 'w')
                for i in range(len(node_g_used_lst)):
                    f.write(str(node_g_used_lst[i]))
                    if i != len(node_g_used_lst) - 1:
                        f.write(',')
                f.close()

                # 存 air 使用量
                path = f"{directories}/node_a_used.txt"
                f = open(path, 'w')
                for i in range(len(node_a_used_lst)):
                    f.write(str(node_a_used_lst[i]))
                    if i != len(node_a_used_lst) - 1:
                        f.write(',')
                f.close()

                # 存 satellite 使用量
                path = f"{directories}/node_s_used.txt"
                f = open(path, 'w')
                for i in range(len(node_s_used_lst)):
                    f.write(str(node_s_used_lst[i]))
                    if i != len(node_s_used_lst) - 1:
                        f.write(',')
                f.close()

                # 存 total 使用量（如果你有用）
                path = f"{directories}/node_used_total.txt"
                f = open(path, 'w')
                for i in range(len(node_used_lst)):
                    f.write(str(node_used_lst[i]))
                    if i != len(node_used_lst) - 1:
                        f.write(',')
                f.close()"""

                # 存 traversal E2E delay
                path = f"{directories}/traversal_E2E_delay.txt"
                f = open(path, 'w')
                for i in range(len(traversal_E2E_delay_lst)):
                    f.write(str(traversal_E2E_delay_lst[i]))
                    if i != len(traversal_E2E_delay_lst) - 1:
                        f.write(',')
                f.close()

            

				# Figure--------------------------------------------------------------
				# Main----------------------------------------------------------------
                DDPGName = 'DDPG'
                directories = f'{datadirectory}/DDPG_{str(MAX_USER)}_figure'
                if not os.path.isdir(directories):
                    os.mkdir(directories)

                plt.figure()
                plt.plot(np.array(p_r_lst_DDPG), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Placement Reward')
                plt.ylabel('Reward')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Placement_Reward.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(m_r_lst_DDPG), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Migration Reward')
                plt.ylabel('Reward')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Migration_Reward.png')
                plt.close()

                """plt.figure()
                plt.plot(np.array(traversal_delay_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Traversal Delay')
                plt.ylabel('Delay (ms)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Traversal_Delay.png')
                plt.close()"""

                plt.figure()
                plt.plot(np.array(migration_service_downtime_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Migration Service Downtime')
                plt.ylabel('Downtime (ms)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Migration_Service_Downtime.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(placement_service_downtime_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Placement Service Downtime')
                plt.ylabel('Downtime (ms)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Placement_Service_Downtime.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(migration_energy_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Migration Energy Consumption')
                plt.ylabel('Energy (J)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Migration_Energy_Consumption.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(placement_energy_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Placement Energy Consumption')
                plt.ylabel('Energy (J)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Placement_Energy_Consumption.png')
                plt.close()


                plt.figure()
                plt.plot(np.array(traversal_energy_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Average traverse energy consumption')
                plt.ylabel('Energy (J)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Average_traverse_energy_consumption.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(PSD_violate_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Placement Violate Rate')
                plt.ylabel('Violation Rate(%)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Placement_Violate_Rate.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(MSD_violate_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Migration Violate Rate')
                plt.ylabel('Violation Rate(%)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Migration_Violate_Rate.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(TD_violate_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Traversal Violate Rate')
                plt.ylabel('Violation Rate(%)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Traversal_Violate_Rate.png')
                plt.close()

                """plt.figure(figsize=(8,6), dpi=300)
                nodes = np.arange(MAX_NODES)
                th_node = []
                for i in range(MAX_NODES):
                    plt.bar(nodes[i], nodes_users_lst[i], width=0.25, color="gray", label=str(i + 1))
                    th_node.append(str(i + 1))
                plt.xticks(nodes, th_node)
                plt.legend()
                plt.ylabel('users')
                plt.xlabel('nodes')
                plt.tight_layout()
                plt.savefig(directories + '/Nodes_Users.png')
                plt.close()

                plt.figure(figsize=(8,6), dpi=300)
                links = np.arange(MAX_LINK)
                th_link = []
                for i in range(MAX_LINK):
                    plt.bar(links[i], links_users_lst[i], width=0.25, color="gray", label=str(i + 1))
                    th_link.append(str(i + 1))
                plt.xticks(links, th_link)
                plt.legend()
                plt.ylabel('users')
                plt.xlabel('links')
                plt.tight_layout()
                plt.savefig(directories + '/Nodes_Links.png')
                plt.close()"""

                plt.figure(figsize=(8,6), dpi=300)
                plt.stackplot(
                    range(len(node_g_used_lst)),  # x 軸直接用 episode index
                    node_g_used_lst,
                    node_a_used_lst,
                    node_s_used_lst,
                    labels=["Ground", "Air", "Satellite"],
                    alpha=0.8
                )

                plt.title(f"Node Assignment Trend (Episode)")
                plt.xlabel("Episode")
                plt.ylabel("Usage Count")
                plt.legend(loc="upper left")
                plt.grid(alpha=0.3)
                plt.tight_layout()

                plt.savefig(directories + '/Nodes_Assignment_Trend.png')
                plt.close()

                plt.figure()
                plt.plot(np.array(traversal_E2E_delay_lst), color = 'blue', label = DDPGName)
                plt.legend()
                plt.title('Traversal E2E Delay')
                plt.ylabel('E2E Delay(ms)')
                plt.xlabel('episode')
                figure = plt.gcf()
                figure.savefig(directories + '/Traversal_E2E_Delay.png')
                plt.close()
    gc.collect() # 強制 Python 釋放未使用的記憶體物件