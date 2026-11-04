import numpy as np
from config import (
    MIN_RESIDENT_TIME,
    MAX_RESIDENT_TIME,
    NUM_SPACE_NODES,
    NUM_AIR_NODES,
    MAX_NODES,
    MAX_SFC_TYPE
)
from typing import List, Optional
from enum import Enum

class EventType(str, Enum):
    PLACEMENT = "placement"
    MIGRATION = "migration"
    RESOURCE  = "resource"
# Placement event = [time, user, res_time, -1, dst_node, SFC type]
# Migration event = [time, user, res_time, src_node, dst_node, SFC type]
# Resource(SFC & migration path) event = [time, user, res_time, dst_node, SFC type, mig_path, mig_path_dir, mig_time, residual_VNF]
class Event:
    def __init__(self, user:int, time:float, res_time:int, dst_node:int, SFC_type:int,
                 event_type:EventType, src_node:Optional[int]=None,
                 mig_path:Optional[List[int]]=None, mig_path_dir:Optional[List[int]]=None,
                 mig_time:Optional[float]=None, residual_VNF:Optional[List[int]]=None,
                 vnf_id:Optional[int]=None):
        self.user = user
        self.time = time
        self.res_time = res_time
        self.src_node = src_node
        self.dst_node = dst_node
        self.SFC_type = SFC_type
        self.event_type = event_type
        self.mig_path = mig_path
        self.mig_path_dir = mig_path_dir
        self.mig_time = mig_time
        self.residual_VNF = residual_VNF
        self.vnf_id = vnf_id

    def is_placement(self): return self.event_type == EventType.PLACEMENT
    def is_migration(self): return self.event_type == EventType.MIGRATION
    def is_resource(self): return self.event_type == EventType.RESOURCE

    def __repr__(self):
        return f"<Event {self.event_type} user={self.user} t={self.time:.2f} src={self.src_node} dst={self.dst_node} res_time={self.res_time} mig_path={self.mig_path} mig_time={self.mig_time}>"

    # ---------- 工廠方法 ----------
    @classmethod
    def new_placement(cls, user:int, avg_dwell_time:float):
        return cls(
            user=user,
            time=np.random.exponential(avg_dwell_time),
            res_time=np.random.randint(MIN_RESIDENT_TIME, MAX_RESIDENT_TIME),
            src_node=-1,
            dst_node=int(np.random.randint(NUM_SPACE_NODES+NUM_AIR_NODES, MAX_NODES)), # 0是LEO, 1~6是無人機, 地面基站是 7~24
            SFC_type=int(np.random.randint(0, MAX_SFC_TYPE)),
            event_type=EventType.PLACEMENT
        )

    @classmethod
    def new_migration(cls, user:int, time:float, res_time:int, src_node:int, dst_node:int, SFC_type:int):
        return cls(
            user=user,
            time=time,
            res_time=res_time,
            src_node=src_node,
            dst_node=dst_node,
            SFC_type=SFC_type,
            event_type=EventType.MIGRATION            
        )

    @classmethod
    def new_resource(cls, user:int, time:float, res_time:int, dst_node:int, SFC_type:int,
                     mig_path:List[int], mig_path_dir:List[int], mig_time:float, residual_VNF:List[int], vnf_id:int=None):
        return cls(
            user=user,
            time=time,
            res_time=res_time,
            dst_node=dst_node,
            SFC_type=SFC_type,
            event_type=EventType.RESOURCE,
            mig_path=mig_path,
            mig_path_dir=mig_path_dir,
            mig_time=mig_time,
            residual_VNF=residual_VNF,
            vnf_id=vnf_id
        )