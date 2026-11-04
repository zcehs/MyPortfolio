import heapq
from collections import defaultdict
from typing import List, Optional
from event import Event

class EventQueue:
    def __init__(self):
        # 全局最小堆，存 (time, event)
        self._event_heap: List[tuple[float, 'Event']] = []
        # 每個 user 的事件列表
        self._user_events: defaultdict[int, List['Event']] = defaultdict(list)

    def add_event(self, event: 'Event'):
        """新增事件到隊列"""
        heapq.heappush(self._event_heap, (event.time, event))
        self._user_events[event.user].append(event)

    def pop_earliest_event(self) -> Optional['Event']:
        """取出全局最早事件"""
        if not self._event_heap:
            return None
        _, event = heapq.heappop(self._event_heap)
        # 從 user 事件列表同步刪除
        self._user_events[event.user].remove(event)
        return event

    def peek_earliest_event(self) -> Optional['Event']:
        """只查看最早事件，不刪除"""
        if not self._event_heap:
            return None
        return self._event_heap[0][1]

    def get_user_events(self, user_id: int, event_type: Optional[str] = None) -> List['Event']:
        """
        取得特定 user 的事件
        event_type 可選："migration", "placement", "resource"，None 表示全取
        """
        events = self._user_events.get(user_id, [])
        if event_type is None:
            return events[:]
        if event_type == "migration":
            return [ev for ev in events if ev.is_migration()]
        elif event_type == "placement":
            return [ev for ev in events if ev.is_placement()]
        elif event_type == "resource":
            return [ev for ev in events if ev.is_resource()]
        else:
            raise ValueError(f"Unknown event_type: {event_type}")

    def remove_event(self, event: 'Event'):
        """手動刪除事件"""
        if event in self._user_events.get(event.user, []):
            self._user_events[event.user].remove(event)
        # 直接從 heap 中刪除會破壞 heap 結構，通常用 pop_earliest_event 取出最早事件即可

    def __len__(self):
        return len(self._event_heap)
