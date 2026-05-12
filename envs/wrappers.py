from __future__ import annotations

from collections import deque

import numpy as np


class ActionDelayBuffer:
    def __init__(self, delay_steps: int, action_dim: int):
        self.delay_steps = max(0, int(delay_steps))
        self.action_dim = int(action_dim)
        self._buffer: deque[np.ndarray] = deque(maxlen=self.delay_steps + 1)
        self.reset()

    def reset(self) -> None:
        self._buffer.clear()
        for _ in range(self.delay_steps + 1):
            self._buffer.append(np.zeros(self.action_dim, dtype=np.float64))

    def push(self, action: np.ndarray) -> np.ndarray:
        self._buffer.append(np.asarray(action, dtype=np.float64).copy())
        return self._buffer.popleft()
