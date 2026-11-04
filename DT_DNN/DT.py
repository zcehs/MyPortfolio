import numpy as np
from config import (
    MIN_DT_SIZE,
    MAX_DT_SIZE,
    MIN_UPDATE_DATA_SIZE,
    MAX_UPDATE_DATA_SIZE,
    DT_PREPROC_RATE,
    MIN_DT_PROC_REQ,
    MAX_DT_PROC_REQ,
    MIN_DT_DR_REQ,
    MAX_DT_DR_REQ,
    MAX_NODES,
)
from typing import List, Optional
from enum import Enum

class DT:
    def __init__(self, DT_id: int, node: int, AoI: float, update_delay : float, DT_size: int, update_size: int, proc_req: int, dr_req: int):
        self.DT_id = DT_id#DT編號
        self.IoT_node = node#DT的IoT資料來源節點
        self.node = node#DT的部署節點
        self.mig_end_time = 0
        self.AoI = AoI#DT的AoI，單位為毫秒
        self.req_count = 0
        self.data_arrival_time = []
        self.update_delay = update_delay
        self.DT_size = DT_size#DT的資料大小，單位為Mbit
        self.update_size = update_size#DT更新所需的資料大小，單位為Mbit
        self.train_data_size = int(DT_size * DT_PREPROC_RATE) #DT預處理後的資料大小
        self.proc_req = proc_req#DT的處理需求，單位為cycles/bit
        self.dr_req = dr_req#DT的資料傳輸需求，單位為bits/s
        self.update_link = None#DT更新所需的傳輸路徑
        self.update_link_dir = None#DT更新所需的傳輸路徑方向
    @classmethod
    def new_DT(cls, DT_id: int):
        return cls(
            DT_id=DT_id,
            node=int(np.random.randint(0, MAX_NODES)),
            AoI=0, #ms
            update_delay=0,
            DT_size=int(np.random.randint(MIN_DT_SIZE, MAX_DT_SIZE)),
            update_size=int(np.random.randint(MIN_UPDATE_DATA_SIZE, MAX_UPDATE_DATA_SIZE)),
            proc_req=int(np.random.randint(MIN_DT_PROC_REQ, MAX_DT_PROC_REQ)), #cycles/bit
            dr_req=int(np.random.randint(MIN_DT_DR_REQ, MAX_DT_DR_REQ)) #bits/s
        )

    def count_AoI(self, AoI):
        self.AoI = ((self.AoI * self.req_count) + AoI) / (self.req_count + 1)
        self.req_count += 1

    def reset_AoI(self):
        self.AoI = 0
        self.req_count = 0