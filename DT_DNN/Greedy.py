import numpy as np
from collections import deque

import env as env_module

from config import (
    DNN_SIZE,
    DNN_RESULT_SIZE_RATE,
)


class Greedy:
    """
    Locality-based Greedy baseline

    DT:
        1. 優先部署在 IoT 所在節點。
        2. 若 IoT 節點 storage 不足，選擇離 IoT hop 數最近的 feasible node。
        3. hop 相同時，選剩餘 storage 較多的節點。

    DNN:
        不使用任何 delay / accuracy / reward 預估。

        對每個 feasible node n 計算：

            score(n)
                = sum(
                    DT.train_data_size * hop(DT.node, n)
                  )
                  + result_size * hop(n, user_node)

        只考慮此次 Event 要求的 DT。

        score 越小越好。
        score 相同時，選剩餘 storage 較多的節點。

    注意：
        - 不呼叫 count_prop_delay()
        - 不呼叫 estimate_prop_delay()
        - 不呼叫 count_proc_delay()
        - 不呼叫 estimate_proc_delay()
        - 不呼叫 count_acc()
        - 不呼叫 get_reward()

    因此這是一個純 topology / data-locality heuristic，
    不知道目前真正的 delay、accuracy 或 reward。
    """

    def __init__(self, env, a_dim, name):
        self.env = env
        self.a_dim = a_dim
        self.name = name

        # 為了和 TD3 / Random 維持相同 interface
        self.pointer = 0
        self.update_cnt = 0

    # =========================================================
    # Common
    # =========================================================
    def choose_action(self, s):
        if self.name == "DT_agent":
            node = self.greedy_DT()
        elif self.name == "DNN_agent":
            node = self.greedy_DNN()
        else:
            raise ValueError(f"Unknown Greedy agent name: {self.name}")

        # ENV 最後用 argmax 選節點，所以輸出 one-hot action
        action = np.zeros((1, self.a_dim), dtype=np.float32)
        action[0][node] = 1.0

        return action

    def _available_storage_for_DT(self, node, dt):
        """
        DT step 真正執行時，ENV.step() 會先 release 原本 DT 的 storage，
        再呼叫 find_max_prior_node()。

        因此若 candidate == 原 DT node，
        Greedy 在 action 選擇階段要把自己的 DT_size 加回去，
        才和 ENV 真正執行時的 feasibility 一致。
        """
        available = self.env.Node_storage[node]

        if node == dt.node:
            available += dt.DT_size

        return available

    def _hop_distance(self, src, dst):
        """
        只根據目前 topology 計算 unweighted shortest-hop distance。

        不讀：
            link_state
            bandwidth
            propagation delay
            CPU state

        main.py 每個 episode 都會更新 env_module.MEC_NODE_GRAPH，
        所以這裡直接讀 env module 裡目前的 topology。
        """
        if src == dst:
            return 0

        graph = env_module.MEC_NODE_GRAPH

        visited = [False for _ in range(self.a_dim)]
        q = deque([(src, 0)])
        visited[src] = True

        while q:
            node, dist = q.popleft()

            for nxt in graph[node]:
                nxt = int(nxt)

                if visited[nxt]:
                    continue

                if nxt == dst:
                    return dist + 1

                visited[nxt] = True
                q.append((nxt, dist + 1))

        return float("inf")

    # =========================================================
    # DT Greedy
    # =========================================================
    def greedy_DT(self):
        env = self.env
        dt = env.DT_lst[env.updating_DT]

        # 如果 migration 尚未完成，ENV.step() 本來就不會套用新 action。
        # 此時回傳目前 node 即可。
        if env.time < dt.mig_end_time:
            return dt.node

        iot_node = dt.IoT_node

        # 1. IoT node 放得下 -> 直接 colocate IoT 與 DT
        iot_available_storage = self._available_storage_for_DT(
            iot_node,
            dt
        )

        if iot_available_storage >= dt.DT_size:
            return iot_node

        # 2. IoT node 放不下 -> 找 hop 最近的 feasible node
        best_node = None
        best_hop = float("inf")
        best_storage = -1

        for node in range(self.a_dim):
            available_storage = self._available_storage_for_DT(
                node,
                dt
            )

            if available_storage < dt.DT_size:
                continue

            hop = self._hop_distance(iot_node, node)

            if np.isinf(hop):
                continue

            if hop < best_hop:
                best_hop = hop
                best_node = node
                best_storage = available_storage

            elif hop == best_hop:
                # hop 相同 -> 剩餘 storage 大的優先
                if available_storage > best_storage:
                    best_node = node
                    best_storage = available_storage

                # storage 也相同 -> node id 小的，讓 baseline deterministic
                elif (
                    available_storage == best_storage
                    and best_node is not None
                    and node < best_node
                ):
                    best_node = node

        if best_node is None:
            raise RuntimeError("Greedy DT: no feasible node")

        return best_node

    # =========================================================
    # DNN Greedy
    # =========================================================
    def greedy_DNN(self):
        env = self.env
        event = env.event

        # 1. 先取得這個 event 真正需要的 DT 與資料量
        requested_dt_ids = []
        total_train_data_size = 0

        for dt_id, is_used in enumerate(event.DT_req_lst):
            if not is_used:
                continue

            requested_dt_ids.append(dt_id)
            total_train_data_size += env.DT_lst[dt_id].train_data_size

        if not requested_dt_ids:
            raise RuntimeError(
                "Greedy DNN: current event does not request any DT"
            )

        # DNN 處理完後傳回 user 的結果資料量
        result_size = total_train_data_size * DNN_RESULT_SIZE_RATE

        # 2. 對每個 feasible node 算 static locality score
        best_node = None
        best_score = float("inf")
        best_storage = -1

        for node in range(self.a_dim):
            # DNN storage constraint
            if env.Node_storage[node] < DNN_SIZE:
                continue

            score = 0.0
            reachable = True

            # DT -> DNN communication locality
            # 資料越大的 DT，希望離 DNN 越近：train_data_size * hop_distance
            for dt_id in requested_dt_ids:
                dt = env.DT_lst[dt_id]

                hop = self._hop_distance(
                    dt.node,
                    node
                )

                if np.isinf(hop):
                    reachable = False
                    break

                score += dt.train_data_size * hop

            if not reachable:
                continue

            # DNN -> User result locality
            user_hop = self._hop_distance(
                node,
                event.dst_node
            )

            if np.isinf(user_hop):
                continue

            score += result_size * user_hop

            # 3. score 最小者勝
            if score < best_score:
                best_score = score
                best_node = node
                best_storage = env.Node_storage[node]

            elif np.isclose(score, best_score):
                # score 一樣 -> storage 多的優先
                if env.Node_storage[node] > best_storage:
                    best_node = node
                    best_storage = env.Node_storage[node]

                # storage 也一樣 -> node id 小的，保持 deterministic
                elif (
                    env.Node_storage[node] == best_storage
                    and best_node is not None
                    and node < best_node
                ):
                    best_node = node

        if best_node is None:
            raise RuntimeError("Greedy DNN: no feasible node")

        return best_node

    # =========================================================
    # Interface compatibility
    # =========================================================
    def initialize_target_networks(self):
        pass

    def store_transition(self, s, a, r, s_):
        pass

    def learn(self):
        pass
