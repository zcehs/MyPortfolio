import numpy as np
from config import (
    NUM_DT,
    MIN_DELAY_CONSTRAINT,
    MAX_DELAY_CONSTRAINT,
    MIN_ACCURACY_REQUIREMENT,
    MAX_ACCURACY_REQUIREMENT,
    MAX_NODES,
    MIN_DR_REQ,
    MAX_DR_REQ,
    MIN_PROC_REQ,
    MAX_PROC_REQ
)
from typing import List, Optional
from enum import Enum

class Event:
    def __init__(self, user:int, time:float, dst_node:int, DT_req_lst: List[bool], delay_cons: int, acc_req: int, dr_req: int, proc_req: int):
        self.user = user #使用者編號
        self.time = time #EVENT發生的時間
        self.dst_node = dst_node #event發生的目的地節點
        self.DT_req_lst = DT_req_lst #event需要的DT類型列表，長度為NUM_DT，每個元素為True或False，表示是否需要該類型的DT
        self.delay_cons = delay_cons #event的延遲要求，單位為毫秒
        self.acc_req = acc_req #event的準確率要求，單位為百分比
        self.dr_req = dr_req #event的資料傳輸要求，單位為bits/s
        self.proc_req = proc_req #event的處理要求，單位為cycles/bit

    @classmethod
    def new_Event(cls, user:int, time:float):
        dt_req_lst = [bool(np.random.randint(0, 2)) for _ in range(NUM_DT)]
        if not any(dt_req_lst):
            selected = int(np.random.randint(0, NUM_DT))
            dt_req_lst[selected] = True
        return cls(
            user=user,
            time=time,
            dst_node=np.random.randint(0, MAX_NODES),
            DT_req_lst = dt_req_lst,
            #delay_cons=np.random.randint(MIN_DELAY_CONSTRAINT, MAX_DELAY_CONSTRAINT),
            delay_cons = MAX_DELAY_CONSTRAINT,
            acc_req = float(np.random.uniform(MIN_ACCURACY_REQUIREMENT, MAX_ACCURACY_REQUIREMENT)),
            dr_req=np.random.randint(MIN_DR_REQ, MAX_DR_REQ),
            proc_req=np.random.randint(MIN_PROC_REQ, MAX_PROC_REQ)
        )