import numpy as np
from config import (
    NUM_DT,
    DNN_SIZE,
    DNN_RESULT_SIZE_RATE
)
from typing import List, Optional
from enum import Enum
class DNN:
    def __init__(self, user: int, time: float, node: int, dst_node: int, delay_cons: float, AoI: float, DT_req_lst: List[bool], dr_req: int, proc_req: int):
        self.user = user#user編號
        self.time = time
        self.res_time = None
        self.node = node
        self.dst_node = dst_node#DNN的部署節點
        self.delay_cons = delay_cons
        self.train_prop_delay_lst = [0 for i in range(NUM_DT)]
        self.train_prop_delay = 0
        self.train_proc_delay = 0
        self.result_prop_delay = 0
        self.result_delay = 0
        self.DNN_size = DNN_SIZE#DT的資料大小，單位為Mbit
        self.AoI = AoI
        self.train_data_size = None
        self.result_size = None #DT預處理後的資料大小
        self.dr_req = dr_req
        self.proc_req = proc_req
        self.DT_req_lst = DT_req_lst.copy() #event需要的DT類型列表，長度為NUM_DT，每個元素為True或False，表示是否需要該類型的DT
        self.pending_DT_lst = DT_req_lst.copy()
        self.train_data_link = [[] for _ in range(NUM_DT)]#DT到DNN所需的傳輸路徑
        self.train_data_link_dir = [[] for _ in range(NUM_DT)]#DT到DNN所需的傳輸路徑
        self.result_link = None#DT到DNN所需的傳輸路徑
        self.result_link_dir = None#DT到DNN所需的傳輸路徑方向,
        self.phase = 0#0:DT to DNN 1:DNN training 2:resturn result
        self.remaining_work = [1 for i in range(NUM_DT)]
    @classmethod
    def new_DNN(cls, user: int, time : float, node: int, dst_node: int, delay_cons: float, DT_req_lst: List[bool], dr_req: int, proc_req: int):
        return cls(
            user=user,
            time = time,
            node=node,
            AoI = 0,
            dst_node=dst_node,
            delay_cons = delay_cons,
            DT_req_lst=DT_req_lst,
            dr_req = dr_req,
            proc_req = proc_req
        )
    
    def set_train_data_size(self, train_data_size):
        self.train_data_size = train_data_size
        self.result_size = int(self.train_data_size * DNN_RESULT_SIZE_RATE) #DT預處理後的資料大小

    def clear_train_data_path(self, count):
        self.pending_DT_lst[count] = False
        self.train_prop_delay_lst[count] = 0
        self.train_data_link[count] = None
        self.train_data_link_dir[count] = None
        self.remaining_work[count] = 0

    """def clear(self):
        self.dst_node = None#DNN的部署節點
        self.DNN_size = DNN_SIZE#DT的資料大小，單位為Mbit
        self.train_data_size = None
        self.result_size = int(self.train_data_size * DNN_RESULT_SIZE_RATE) #DT預處理後的資料大小
        self.DT_req_lst = None #event需要的DT類型列表，長度為NUM_DT，每個元素為True或False，表示是否需要該類型的DT
        self.train_data_link = None#DT到DNN所需的傳輸路徑
        self.train_data_link_dir = None#DT到DNN所需的傳輸路徑方向
        #self.phase=0"""