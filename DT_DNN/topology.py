import json
import random
import matplotlib.pyplot as plt
import networkx as nx
import os

path = "topology/"

if not os.path.isdir('topology'):
    os.mkdir('topology')
    
ms_per_km = 0.005 # ms


MEC_min_dis = 5 # km 
MEC_max_dis = 6 # km

def save_data(directory, data, data_name):
    """將完整資料以 JSON 格式存入 txt。"""
    file_path = os.path.join(directory, f"{data_name}.txt")

    with open(file_path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False)

def generate_graph(MAX_NODE, p, seed):
    link_num = 0

    MEC_Node_Graph = []
    MEC_Link_Graph = []
    Link_propagation = []

    G = nx.erdos_renyi_graph(MAX_NODE, p, seed = seed)
    pos = nx.spring_layout(G, seed = seed)

    n_labels = {}
    e_labels = {}
    e_i_labels = {}

    for i in range(MAX_NODE):
        n_labels[i] = str(i)
        key_lst = [k for k in G.adj[i]]
        MEC_Node_Graph.append(sorted(key_lst))

        # MEC_Link_Graph.append()
        for j in key_lst:
            if e_labels.get((j, i), None) == None:
                e_labels[(i, j)] = round((MEC_min_dis + (MEC_max_dis - MEC_min_dis) * random.random()) *  ms_per_km, 3) # 0.025~0.03 ms
                Link_propagation.append(e_labels[(i, j)])
                e_i_labels[(i, j)] = link_num # link index
                link_num += 1
        
        temp_lst = []
        for j in MEC_Node_Graph[i]:
            if e_i_labels.get((i, j), None) != None:
                temp_lst.append(e_i_labels[(i, j)])
            elif e_i_labels.get((j, i), None) != None:
                temp_lst.append(e_i_labels[(j, i)])
        
        MEC_Link_Graph.append(temp_lst)


    return MEC_Node_Graph, MEC_Link_Graph, Link_propagation, link_num


MEC_Node_Graph_lst = []
MEC_Link_Graph_lst = []
Link_propagation_lst = []
link_num_lst = []

MEC_Node_Graph = 0
MEC_Link_Graph = 0
Link_propagation = 0
link_num = 0

for i in range(2000):

    MEC_Node_Graph, MEC_Link_Graph, Link_propagation, link_num = generate_graph(18, 0.2, 38)
    MEC_Node_Graph_lst.append(MEC_Node_Graph)
    MEC_Link_Graph_lst.append(MEC_Link_Graph)
    Link_propagation_lst.append(Link_propagation)
    link_num_lst.append(link_num)

save_data(path, MEC_Node_Graph_lst, 'Node_Graph')
save_data(path, MEC_Link_Graph_lst, 'Link_Graph')
save_data(path, Link_propagation_lst, 'Propagation_Delay')
save_data(path, link_num_lst, 'Link_Num')

print("Graph generation completed. Data saved in 'topology' directory.")

