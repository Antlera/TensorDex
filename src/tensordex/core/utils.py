import time
from collections import defaultdict


class TensorDexTimer:
    def __init__(self):
        self._start_time = {}
        self._elapsed_time = defaultdict(float)
        self._recent_time = {}

    def start(self, name):
        if name in self._start_time:
            raise RuntimeError(f"Timer '{name}' is already running.")
        self._start_time[name] = time.perf_counter()

    def stop(self, name):
        if name not in self._start_time:
            raise RuntimeError(f"Timer '{name}' was not started.")
        elapsed = time.perf_counter() - self._start_time.pop(name)
        self._elapsed_time[name] += elapsed
        self._recent_time[name] = elapsed
        return elapsed

    def get(self, name) -> float:
        return self._elapsed_time.get(name, 0.0)

    def get_recent(self, name) -> float:
        return self._recent_time.get(name, 0.0)

    def reset(self, name=None):
        if name:
            self._start_time.pop(name, None)
            self._elapsed_time.pop(name, None)
            self._recent_time.pop(name, None)
        else:
            self._start_time.clear()
            self._elapsed_time.clear()
            self._recent_time.clear()

    def report(self):
        return dict(self._elapsed_time)

    def report_recent(self):
        return dict(self._recent_time)
