import matplotlib.pyplot as plt
import networkx as nx
import os
import numpy as np
import math


path = "./topology/"
if not os.path.exists(path):
    os.makedirs(path)

np.random.seed(10)

# SAGIN
node_s = 1
node_a = 6
node_g = 18
group_set = 3
max_node = node_s + node_a + node_g

s_g_dis = 500 #km
a_g_dis = 10 #km
s_a_dis = s_g_dis - a_g_dis

a_a_dis = 10 #km
g_g_dis = 7.8375
g_g_dis_group = 5 #km

air_y = 10000 #km
node_location = [[], 
                 [-5000, 5000 * math.sqrt(3), air_y], [15000, 0, air_y], [5000, -5000 * math.sqrt(3), air_y], [-5000, -5000 * math.sqrt(3), air_y], [-15000, 0, air_y], [5000, 5000 * math.sqrt(3), air_y]]

def cot(x):
    return 1 / math.tan(x)

def ses(x):
    return 1 / math.sin(x)

def generate_graph():
    # const
    ms_per_km = 0.0033 # ms
    # ms_per_km = 1/ 299792/1000 # ms

    # seed = 38
    # random.seed(seed)

    #set x y z pos for each node
    for i in range(node_s, node_s + node_a):
        n0_location = [node_location[i][0], node_location[i][1] + 5000 * math.sqrt(3) / 3, 0]
        n1_location = [node_location[i][0] - 5000 / 2, node_location[i][1] - 5000 * math.sqrt(3) / 6, 0]
        n2_location = [node_location[i][0] + 5000 / 2, node_location[i][1] - 5000 * math.sqrt(3) / 6, 0]
        node_location.append(n0_location)
        node_location.append(n1_location)
        node_location.append(n2_location)

    # node #0:satellite #1~3:UAV #4~8:BS
    node = [x for x in range(max_node)]
    air_node = [x for x in range(node_s, node_s + node_a)]
    ground_node = [[x * group_set + y + node_s + node_a for y in range(group_set)] for x in range(node_g // group_set)]
    # n_labels ={i:str(i)  for i in range(len(node))}

    G  = nx.Graph()
    G.add_nodes_from(node)

    edge = []
    #衛星為全連接的edge
    for i in range(node_s):
        for j in range(i + 1, max_node):
            edge.append((node[i], node[j]))
    
    #UAV的edge
    for i in range(node_a):
        edge.append((node[air_node[node_a - i - 1]], node[air_node[node_a - i - 2]]))#HAP和HAP的edge
        for j in range(group_set):
            edge.append((node[air_node[node_a - i - 1]], node[ground_node[node_a - i - 1][j]]))#HAP和地面的edge

    
    #地面的edge
    for i in range(node_g // group_set):
        edge.append((node[ground_node[node_g // group_set - i - 1][0]], node[ground_node[node_g // group_set - i - 2][-1]]))
        for j in range(group_set):
            edge.append((node[ground_node[i][group_set - j - 1]], node[ground_node[i][group_set - j - 2]]))
        

        
    
    G.add_edges_from(edge)

    n_labels = {}
    e_labels = {}
    e_i_labels = {}
    pos = {}
    MEC_Node_Graph = []
    MEC_Link_Graph = []
    Link_propagation = []
    Link_arr = []
    share_link = []
    link_num = 0
    current_row = -1
    current_col = 0
    row_first_node = 0
    edge_type = [[],[],[],[],[]] # 衛星和HAP、衛星和地面、HAP和HAP、HAP和地面、地面和地面
    for i in range(max_node):
        n_labels[i] = str(i)
        key_lst = [k for k in G.adj[i]]
        MEC_Node_Graph.append(sorted(key_lst))
        
        if i < node_s:
            current_row = 2 * 4
            current_col = node_g * 2 / 2
        elif i < node_s + node_a:
            current_row = 1 * 4
            current_col = 2 * group_set * (i - node_s) + 2
        else:
            current_row = 0 * 4
            current_col = 2 * (i - node_s - node_a)
        pos[i] = (current_col, current_row)
                
        for j in key_lst:
            if e_labels.get((j, i), None) == None:
                if i < node_s:
                    if j<node_s:
                        print('只有一顆衛星')
                    elif j >= node_s and j < node_s+node_a: #衛星和HAP
                        edge_type[0].append(link_num)
                        e_labels[(i, j)] = round( s_a_dis *  ms_per_km, 3) # 490*0.0033 = 1.617
                    elif j >= node_s+node_a and j < max_node: #衛星和地面
                        edge_type[1].append(link_num)
                        e_labels[(i, j)] = round( s_g_dis *  ms_per_km, 3) # 500*0.0033 = 1.65
                elif i < node_s + node_a:
                    if j<node_s: #HAP和衛星 (上個if已經進去)
                        e_labels[(i, j)] = round( s_a_dis *  ms_per_km, 3) # 490*0.0033 = 1.617
                    elif j >= node_s and j < node_s+node_a: # HAP和HAP
                        edge_type[2].append(link_num)
                        e_labels[(i, j)] = round( a_a_dis *  ms_per_km, 3) # 10*0.0033 = 0.033
                    elif j >= node_s+node_a and j < max_node: # HAP和地面
                        edge_type[3].append(link_num)
                        e_labels[(i, j)] = round( a_g_dis *  ms_per_km, 3) # 10*0.0033 = 0.033
                else:
                    if j<node_s: #地面和衛星 (上上個if已經進去)
                        e_labels[(i, j)] = round( s_g_dis *  ms_per_km, 3) # 500*0.0033 = 1.65
                    elif j >= node_s and j < node_s+node_a: #地面和HAP (上上個if已經進去)
                        e_labels[(i, j)] = round( a_g_dis *  ms_per_km, 3) # 10*0.0033 = 0.033
                    elif j >= node_s+node_a and j < max_node: #地面和地面
                        edge_type[4].append(link_num)
                        if abs(i-j)>2: #內圈
                            e_labels[(i, j)] = round( g_g_dis *  0.005, 3) # 5*0.0033 = 0.0258
                        else: #同一個group, 同一個無人機管轄
                            e_labels[(i, j)] = round( g_g_dis_group *  0.005, 3) # 5*0.0033 = 0.0165
                       
                
                # print("e_labels[(i, j)] = ",e_labels[(i, j)])
                Link_propagation.append(e_labels[(i, j)])
                Link_arr.append([i,j])
                e_i_labels[(i, j)] = link_num # link index
                link_num += 1
        
        temp_lst = []
        for j in MEC_Node_Graph[i]:
            if e_i_labels.get((i, j), None) != None:
                temp_lst.append(e_i_labels[(i, j)])
            elif e_i_labels.get((j, i), None) != None:
                temp_lst.append(e_i_labels[(j, i)])
        MEC_Link_Graph.append(temp_lst)
        
    # print(pos)
    for i in range(node_s+node_a):
        share_link.append([])
        for j in range(len( MEC_Link_Graph[i])):
            check = True
            for k in range(len(share_link)):
                if MEC_Link_Graph[i][j] in share_link[k]:
                    check = False
                    break
            if check:
                share_link[i].append(MEC_Link_Graph[i][j])



    name = "Network_weighted_graph"
    plt.figure(figsize=(60, 60))

    edge_diff = [[node_s, node_s + node_a - 1]] + [[x * group_set + node_s + node_a, x * group_set + node_s + node_a + group_set - 1] for x in range(node_g // group_set)] + [[node_s + node_a, max_node -1]]
    # 如果要對不同邊使用不同顏色或寬度，也可以建 dict
    """for u, v in G.edges():
        print(f"bending edge {u}-{v}")"""

    for u, v in G.edges():
    # 先找 (u,v) 或 (v,u) 的設定（無設定則 rad=0）
        x_move = 0
        if [u, v] in edge_diff:
            rad = 0.3   
            dis_x = ((pos[u][0] - pos[v][0]) / 2) # 弓形直邊長度/2
            x_move = ses(rad / 2) * dis_x - cot(rad / 2) * dis_x# 半徑 - 弓形直邊到原點距離
            #print(f"edge {u}-{v} use rad {rad} with x_move {x_move}")
        else:
            rad = 0.0
        nx.draw_networkx_edges(
            G, pos,
            edgelist=[(u, v)],
            connectionstyle=f"arc3,rad={rad}",
            edge_color="black",
            width=2
        )
        lx = (pos[u][0] + pos[v][0]) / 2 
        ly = (pos[u][1] + pos[v][1]) / 2 + x_move


        label_text = str(e_labels.get((u, v), e_labels.get((v, u), "")))
        plt.text(lx, ly, label_text,
                 fontsize=20,
                 horizontalalignment='center',
                 verticalalignment='center',
                 bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.3))
        
 
    nx.draw_networkx_nodes(G, pos, node_size=3000)

    nx.draw_networkx_labels(G, pos, n_labels, font_size=20, font_color="whitesmoke", font_weight="bold")
    #nx.draw_networkx_edge_labels(G, pos, e_labels, font_size=10, verticalalignment = "bottom", rotate=False)
    # adjust_text(e_labels, )
    figure = plt.gcf()
    figure.savefig(path + name)

    
    return MEC_Node_Graph, MEC_Link_Graph, Link_propagation, link_num, share_link, Link_arr, edge_type

#MEC_Node_Graph, MEC_Link_Graph, Link_propagation , MAX_LINK, share_link, Link_arr, edge_type = generate_graph()

#print(MEC_Node_Graph)
