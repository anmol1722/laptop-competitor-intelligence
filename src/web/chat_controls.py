"""Bounded, in-process chat quotas and answer caching for a single web worker."""

from __future__ import annotations

import copy
import math
import threading
import time
from collections import OrderedDict, deque
from typing import Any, Callable


class ChatControls:
    def __init__(self, *, limits: tuple[tuple[int, int], ...] = ((3, 60), (20, 3600)),
                 cache_ttl: int = 600, cache_size: int = 128,
                 max_clients: int = 4096, clock: Callable[[], float] = time.monotonic):
        self.limits = limits
        self.cache_ttl = cache_ttl
        self.cache_size = cache_size
        self.max_clients = max_clients
        self.clock = clock
        self._clients: OrderedDict[str, deque[float]] = OrderedDict()
        self._cache: OrderedDict[tuple, tuple[float, dict[str, Any]]] = OrderedDict()
        self._cooldown_until = 0.0
        self._lock = threading.Lock()

    def check_rate(self, client: str) -> int:
        """Consume one valid request, or return seconds until the client can retry."""
        now = self.clock()
        with self._lock:
            events = self._clients.pop(client, deque())
            longest = max(window for _, window in self.limits)
            while events and events[0] <= now - longest:
                events.popleft()
            wait = 0.0
            for limit, window in self.limits:
                recent = [event for event in events if event > now - window]
                if len(recent) >= limit:
                    wait = max(wait, recent[-limit] + window - now)
            if not wait:
                events.append(now)
            self._clients[client] = events
            while len(self._clients) > self.max_clients:
                self._clients.popitem(last=False)
            return math.ceil(wait)

    def get_cached(self, key: tuple) -> dict[str, Any] | None:
        with self._lock:
            item = self._cache.pop(key, None)
            if item is None or self.clock() >= item[0]:
                return None
            self._cache[key] = item
            return copy.deepcopy(item[1])

    def cache_answer(self, key: tuple, payload: dict[str, Any]) -> None:
        with self._lock:
            self._cache[key] = (self.clock() + self.cache_ttl, copy.deepcopy(payload))
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def defer_provider(self, seconds: int) -> None:
        with self._lock:
            self._cooldown_until = max(self._cooldown_until, self.clock() + seconds)

    def retry_after(self) -> int:
        with self._lock:
            return max(0, math.ceil(self._cooldown_until - self.clock()))
