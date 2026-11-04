class Resource_Event:
    def __init__(self, user:int, time:float, mig_num:int, size:int, res_time:int, proc_req: int, dr_req: int, link, link_dir):
        self.user = user #使用者編號
        self.time = time #EVENT發生的時間
        self.mig_num = mig_num #event發生的目的地節點
        self.size = size #event所需的資源大小，單位為bits
        self.res_time = res_time #event結束時間
        self.proc_req = proc_req #event的處理要求，單位為cycles/bit
        self.dr_req = dr_req #event的資料傳輸要求，單位為bits/s
        self.link = link #event所需的傳輸路徑
        self.link_dir = link_dir #event所需的傳輸路徑方向
        self.remaining_work = 1