# SPDX-License-Identifier: Apache-2.0
"""Bounded memoization of spatial graph outputs during one fixed-weight interval."""
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from .device_events import staged_spatial_key


_ACTIVE = ContextVar("brass_fixed_board_cache", default=None)


def active_cache():
    return _ACTIVE.get()


@contextmanager
def fixed_boards():
    """Each collection or sequence owns a cache. No entry crosses an Adam boundary."""
    cache = BoardCache(2 * 1024**2)
    token = _ACTIVE.set(cache)
    try:
        yield cache
    finally:
        _ACTIVE.reset(token)


class BoardCache:
    def __init__(self, capacity):
        self.capacity = self.limit = capacity
        self.used = 0
        self.entries = OrderedDict()
        self.models = {}

    def limit_to(self, available):
        self.limit = min(self.capacity, max(0, available))
        self._evict(self.limit)

    def _evict(self, maximum):
        while self.used > maximum:
            _, (_, size) = self.entries.popitem(last=False)
            self.used -= size

    def key(self, model, public):
        if self.limit == 0:
            return None
        raw = isinstance(public.locations, np.ndarray) and public.locations.ndim == 2
        if not raw:
            spatial = staged_spatial_key(public)
            if spatial is None:
                return None
        # A model reference distinguishes frozen opponents. The caller owns each
        # scope synchronously, with no step or device transfer inside a forward.
        if model not in self.models:
            self.models[model] = tuple(model.parameters()) + tuple(model.buffers())
        try:
            versions = tuple(v._version for v in self.models[model])
        except RuntimeError:  # Tensors created in inference_mode lack version counters.
            return None
        if raw:
            arrays = (public.locations, public.slots, public.links, public.merchants)
            if any(not isinstance(v, np.ndarray) for v in arrays):
                return None
            spatial = tuple((v.dtype.str, v.shape, v.tobytes()) for v in arrays)
        return model, model.config.board_encoder, versions, spatial

    def get(self, key):
        if key not in self.entries:
            return None
        self.entries.move_to_end(key)
        return self.entries[key][0]

    def store(self, key, context, tables, width):
        spatial = {k: tables[k] for k in ("location", "slot", "link", "merchant")}
        # Charge the full live tensor storage, exact feature bytes and generous
        # Python/key/array bookkeeping. The board slice gets separate storage.
        size = 4096 + sum(len(field[2]) for field in key[-1]) + 4 * width * context.element_size()
        size += sum(t.untyped_storage().nbytes() for t in spatial.values())
        if size > self.limit:
            return
        self._evict(self.limit - size)
        board = context[-4*width:].clone()
        self.entries[key] = ((spatial, board), size)
        self.used += size
