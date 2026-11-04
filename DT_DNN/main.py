import numpy as np
import math
import os
import matplotlib.pyplot as plt
import config
import env as env_module
from env import ENV
import time
from pathlib import Path
from TD3 import TD3
from Random import Random
from Greedy import Greedy
from compare import compare_all
import tensorflow.compat.v1 as tf
from config import(
    MAX_EPISODES,
    MAX_EP_STEPS,
    MAX_INT,
    MAX_USER,
    METHOD,
    GAMMA_DT,
    LR_A_DT,
    LR_C_DT,
    TAU_DT,
    MEMORY_CAPACITY_DT,
    START_LEARNING_DT,
    BATCH_SIZE_DT,
    GAMMA_DNN,
    LR_A_DNN,
    LR_C_DNN,
    TAU_DNN,
    MEMORY_CAPACITY_DNN,
    START_LEARNING_DNN,
    BATCH_SIZE_DNN,
    NUM_DT,
    MAX_TIME,
    get_graph_info
)

def save_data(dir, method, users, data_lst, data_name):
    path = f'{dir}/{method}_{data_name}_{str(users)}.txt'
    f = open(path, 'w')
    for i in range(len(data_lst)):
        f.write(str(data_lst[i]))
        if i != len(data_lst) - 1:
            f.write(',')
    f.close()



def save_figure(dir, method, users, data_lst, data_name):
    save_dir = Path(dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    save_path = save_dir / f"{method}_{data_name}_{users}.png"

    plt.figure()
    plt.plot(np.array(data_lst), color='blue', label=method)
    plt.legend()
    plt.title(data_name)
    plt.ylabel(data_name)
    plt.xlabel('episode')

    figure = plt.gcf()

    try:
        figure.savefig(str(save_path), dpi=150, bbox_inches='tight')

    except OSError as e:
        print("\n========== SAVE FIGURE ERROR ==========")
        print("path:", repr(str(save_path)))
        print("exists:", save_path.exists())
        print("is_file:", save_path.is_file())
        print("is_dir:", save_path.is_dir())
        print("parent exists:", save_path.parent.exists())
        print("error:", repr(e))
        print("=======================================\n")

        # 如果原本檔案被卡住，改存另一個名稱避免整個訓練中斷
        backup_path = save_dir / f"{method}_{data_name}_{users}_backup_{int(time.time())}.png"

        print("Try backup path:", backup_path)
        figure.savefig(str(backup_path), dpi=150, bbox_inches='tight')

    finally:
        plt.close()

def choose_method(method, env, a_dim, s_dim, a_bound, name, sess, memory_capacity, gamma, lr_a, lr_c, tau, batch_size):
    if method == 'TD3':
        return TD3(a_dim, s_dim, a_bound, name, sess, memory_capacity, gamma, lr_a, lr_c, tau, batch_size)
    elif method == 'Random':
        return Random(a_dim)
    elif method == 'Greedy':
        return Greedy(env, a_dim, name)



localtime = time.localtime()
print(time.strftime("%Y-%m-%d %H:%M:%S", localtime))
if not os.path.isdir('trainData'):
    os.mkdir('trainData')

datadirectory = 'trainData/' + time.strftime("%m%d_%H%M%S", localtime)
if not os.path.isdir(datadirectory):
    os.mkdir(datadirectory)

for u, users in enumerate(MAX_USER):
    for method in METHOD:
        log_directories = f'{datadirectory}/{method}_{str(users)}_log'
        if not os.path.isdir(log_directories):
            os.mkdir(log_directories)

        fig_directories = f'{datadirectory}/{method}_{str(users)}_figure'
        if not os.path.isdir(fig_directories):
            os.mkdir(fig_directories)


        print("environment user:", str(users))
        print("MAX_EPISODES:", str(MAX_EPISODES))

        ###############################  training  ####################################
        MAX_EP = MAX_EPISODES
        if method != 'TD3':
            MAX_EP = 200
        env = ENV(users)

        a_dim = env.a_dim
        a_bound = env.a_bound

        tf.reset_default_graph()

        sess = tf.Session()

        DT_agent = choose_method(method, env, a_dim=a_dim, s_dim=env.dt_s_dim, a_bound=a_bound, name="DT_agent", sess=sess, 
                                 memory_capacity=MEMORY_CAPACITY_DT, gamma = GAMMA_DT, lr_a=LR_A_DT, lr_c=LR_C_DT, tau=TAU_DT, batch_size=BATCH_SIZE_DT)

        DNN_agent = choose_method(method, env, a_dim=a_dim, s_dim=env.dnn_s_dim, a_bound=a_bound, name="DNN_agent", sess=sess, 
                                  memory_capacity=MEMORY_CAPACITY_DNN * users, gamma = GAMMA_DNN, lr_a=LR_A_DNN, lr_c=LR_C_DNN, tau=TAU_DNN, batch_size=BATCH_SIZE_DNN)

        # 兩個 Agent 都建立完成後，統一初始化
        sess.run(tf.global_variables_initializer())

        # 將 eval network 權重複製給 target network
        DT_agent.initialize_target_networks()
        DNN_agent.initialize_target_networks()

        # 蒐集輸出資料----------------------------------------------------------------
        DT_r_lst = []
        DNN_r_lst = []

        update_delay_lst = [[] for i in range(NUM_DT)]
        Average_update_delay_lst = []
        AoI_lst = [[] for i in range(NUM_DT)]
        Average_AoI_lst = []
        AoI_by_step_lst = [[] for i in range(NUM_DT)]
        Average_AoI_by_step_lst = []

        service_downtime_lst = []
        estimated_result_delay_lst = []
        estimated_acc_lst = []

        denined_count_lst = []

        #----------------------------------------------------------------
        var_DT = 0.2  # control exploration
        var_DNN = 0.2
        for i in range(MAX_EP):

            DT_reward = 0
            DNN_reward = 0

            DT_event_num = 0
            DT_applied_num = 0
            DNN_event_num = 0

            update_delay = [0 for i in range(NUM_DT)]
            AoI = [0 for i in range(NUM_DT)]
            AoI_by_step = [0 for i in range(NUM_DT)]

            service_downtime = 0
            estimated_result_delay = 0
            estimated_acc = 0
            denined_count = 0

            get_graph_info(i)

            # 將 config 中該 episode 的 topology 同步到 env.py
            env_module.MEC_NODE_GRAPH = config.MEC_NODE_GRAPH
            env_module.MEC_LINK_GRAPH = config.MEC_LINK_GRAPH
            env_module.LINK_PROPAGATION = config.LINK_PROPAGATION
            env_module.MAX_LINK = config.MAX_LINK

            DT_pre_state = [None for _ in range(NUM_DT)]
            DT_pre_act = [None for _ in range(NUM_DT)]
            DT_r = [0 for _ in range(NUM_DT)]
            DT_r_count = [0 for _ in range(NUM_DT)]
    
            DNN_pre_state = None
            DNN_pre_act = None
            DNN_pre_reward = None

            env.reset()

            while (env.time < MAX_TIME):
                #print("Episode:", i, "Step:", j, "updating:", env.updating, "updating_DT:", env.updating_DT)
                """if math.isnan(os[0][0]):
                    os._exit()"""

                if(env.updating):
                    for l in range(NUM_DT):
                        if DT_pre_state[l] is not None:
                            DT_r[l] += env.get_reward(0, l)
                            DT_r_count[l] += 1
                    for l in range(NUM_DT):
                        s = env.get_state(0)
                        if DT_pre_state[l] is not None:
                            AoI[l] += env.get_average_AoI(l)
                        
                        a = DT_agent.choose_action(s)
                        if method == 'TD3':
                            a = np.clip(np.random.normal(a, var_DT), 1e-2, 1 - 1e-2)    # add randomness to action selection for exploration
                        
                        _, action_applied = env.step(a)
                        if action_applied:
                            if DT_pre_state[l] is not None:
                                #DT_r[l] = DT_r[l] / DT_r_count[l]
                                # update DT_agent
                                DT_agent.store_transition(DT_pre_state[l], DT_pre_act[l], DT_r[l], s)
                                DT_reward += DT_r[l]
                                DT_pre_state[l] = s
                                DT_pre_act[l] = a
                                DT_r[l] = 0
                                DT_r_count[l] = 0
                            DT_pre_state[l] = s
                            DT_pre_act[l] = a
                    

                        if DT_agent.pointer > START_LEARNING_DT:
                            #var_DT = max(var_DT * 0.9995, 0.05)
                            DT_agent.learn()

                        ##########   Used for output   ##########
                        if action_applied:
                            DT_applied_num += 1
                        DT_event_num += 1
                    for l in range(NUM_DT):
                        update_delay[l] += env.get_update_delay(l)
                    

                else:
                    s = env.get_state(1)
                    if DNN_pre_state is not None:
                        #r = env.get_reward(1)
                        # update DNN_agent
                        DNN_agent.store_transition(DNN_pre_state, DNN_pre_act, DNN_pre_reward, s)
                        DNN_reward += DNN_pre_reward
                    a = DNN_agent.choose_action(s)
                    if method == 'TD3':
                        a = np.clip(np.random.normal(a, var_DNN), 1e-2, 1 - 1e-2)
                    DNN_pre_state = s
                    DNN_pre_act = a

                    DNN_pre_reward, action_applied = env.step(a)

                    #env.get_reward(1)

                    if DNN_agent.pointer > START_LEARNING_DNN * users:
                        #var_DNN = max(var_DNN * 0.9995, 0.05)
                        DNN_agent.learn()

                    ##########   Used for output   ##########
                    DNN_event_num += 1
                    estimated_acc += env.estimated_acc

            """s = env.get_state(0)
            if DT_pre_state is not None:
                r = env.get_reward(0)
                # update DT_agent
                DT_agent.store_transition(DT_pre_state, DT_pre_act, r, s)
                DT_reward += r

            s = env.get_state(1)
            if DNN_pre_state is not None:
                #r = env.get_reward(1)
                # update DNN_agent
                DNN_agent.store_transition(DNN_pre_state, DNN_pre_act, DNN_pre_reward, s)
                DNN_reward += DNN_pre_reward"""

            if DT_agent.pointer > START_LEARNING_DT:
                var_DT = max(var_DT * 0.995, 0)

            if DNN_agent.pointer > START_LEARNING_DNN:
                var_DNN = max(var_DNN * 0.995, 0)


            DT_reward = DT_reward / DT_applied_num
            DNN_reward = DNN_reward / DNN_event_num
            DT_r_lst.append(DT_reward)
            DNN_r_lst.append(DNN_reward)

            average_update_delay = 0
            average_AoI = 0
            average_AoI_by_step = 0
            for k in range(NUM_DT):
                count = DT_event_num // NUM_DT
                if k < DT_event_num % NUM_DT:
                    count += 1
                update_delay[k] = update_delay[k] / count
                average_update_delay += update_delay[k]
                update_delay_lst[k].append(update_delay[k])

                AoI[k] = AoI[k] / count
                average_AoI += AoI[k]
                AoI_lst[k].append(AoI[k])

                AoI_by_step[k] = env.get_average_AoI_by_step(k)
                AoI_by_step_lst[k].append(AoI_by_step[k])
                average_AoI_by_step += AoI_by_step[k]

            Average_update_delay_lst.append(average_update_delay / NUM_DT)
            Average_AoI_lst.append(average_AoI / NUM_DT)
            Average_AoI_by_step_lst.append(average_AoI_by_step / NUM_DT)

            service_downtime = env.service_downtime
            service_downtime = (service_downtime / env.finished_DNN_count)
            service_downtime_lst.append(service_downtime)

            estimated_result_delay = env.estimated_result_delay
            estimated_result_delay = (estimated_result_delay / DNN_event_num)
            estimated_result_delay_lst.append(estimated_result_delay)

            estimated_acc = estimated_acc / DNN_event_num
            estimated_acc_lst.append(estimated_acc)

            denined_count = env.denined_DNN
            denined_count_lst.append(denined_count)



            print()
            print("Episode = ",i)
            print("----------------------------------------------------")
            print("DT steps = ", DT_event_num)
            print("DNN steps = ", DNN_event_num)
            print("----------------------------------------------------")
            print("DT reward = ", DT_reward)
            print("DNN reward = ", DNN_reward)
            print("----------------------------------------------------")
            print("Average update delay of DTs = ", update_delay)
            print("Average AoI of DTs = ", AoI)
            print("Average AoI by step of DTs = ", AoI_by_step)
            print("Average estimated result delay of DNNs = ", estimated_result_delay)
            print("Average service downtime of DNNs = ", service_downtime)
            print("Average Accuracy of DNNs = ", estimated_acc)
            print("----------------------------------------------------")
            print("Denined DNN = ", denined_count)
            """print("----------------------------------------------------")
            print("DT pointer =", DT_agent.pointer)
            print("DNN pointer =", DNN_agent.pointer)

            print("DT update count =", DT_agent.update_cnt)
            print("DNN update count =", DNN_agent.update_cnt)

            print("DT noise =", var_DT)
            print("DNN noise =", var_DNN)"""

            # 輸出log--------------------------------------------------------------
            if i % 50 == 49:
                # Log--------------------------------------------------------------
                save_data(log_directories, method, users, DT_r_lst, 'DT_reward')

                save_data(log_directories, method, users, DNN_r_lst, 'DNN_reward')

                for k in range(NUM_DT):
                    save_data(log_directories, method, users, update_delay_lst[k], f'update_delay_DT{k}')

                save_data(log_directories, method, users, Average_update_delay_lst, 'Average_update_delay_DT')

                for k in range(NUM_DT):
                    save_data(log_directories, method, users, AoI_lst[k], f'AoI{k}')

                save_data(log_directories, method, users, Average_AoI_lst, 'Average_AoI_DT')

                for k in range(NUM_DT):
                    save_data(log_directories, method, users, AoI_by_step_lst[k], f'AoI_by_step{k}')

                save_data(log_directories, method, users, Average_AoI_by_step_lst, 'Average_AoI_by_step_lst')

                save_data(log_directories, method, users, service_downtime_lst, 'service_downtime')

                save_data(log_directories, method, users, estimated_result_delay_lst, 'estimated_result_delay')

                save_data(log_directories, method, users, estimated_acc_lst, 'estimated_Accuracy')

                #save_data(log_directories, method, users, denined_count_lst, "denined_events")

                # Figure--------------------------------------------------------------
                save_figure(fig_directories, method, users, DT_r_lst, 'DT_reward')

                save_figure(fig_directories, method, users, DNN_r_lst, 'DNN_reward')

                for k in range(NUM_DT):
                    save_figure(fig_directories, method, users, update_delay_lst[k], f'update_delay_DT{k}')

                save_figure(fig_directories, method, users, Average_update_delay_lst, 'Average_update_delay_DT')

                for k in range(NUM_DT):
                    save_figure(fig_directories, method, users, AoI_lst[k], f'AoI{k}')

                save_figure(fig_directories, method, users, Average_AoI_lst, 'Average_AoI_DT')

                for k in range(NUM_DT):
                    save_figure(fig_directories, method, users, AoI_by_step_lst[k], f'AoI_by_step{k}')

                save_figure(fig_directories, method, users, Average_AoI_by_step_lst, 'Average_AoI_by_step_lst')

                save_figure(fig_directories, method, users, service_downtime_lst, 'service_downtime')

                save_figure(fig_directories, method, users, estimated_result_delay_lst, 'estimated_result_delay')

                save_figure(fig_directories, method, users, estimated_acc_lst, 'estimated_Accuracy')

                #save_figure(fig_directories, method, users, denined_count_lst, "denined_events")

# ============================================================================
# 所有壓力測試 (MAX_USER) 與所有方法 (METHOD) 都跑完後，自動產生比較結果
# 注意：這段一定要放在兩層 for users / for method 迴圈外面。
# ============================================================================
    compare_all(
        data_path=datadirectory,
        methods=METHOD,
        user_counts=MAX_USER[0: u + 1],
        last_n=min(200, MAX_EPISODES),  # 預設比較最後 200 episodes
    )
