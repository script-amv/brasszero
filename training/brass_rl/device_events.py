# SPDX-License-Identifier: Apache-2.0
"""Update-local device placement of raw event fields; all model math stays scalar."""
from dataclasses import dataclass, fields, replace

import numpy as np
import torch

from .features import EncodedCandidates, EncodedPublic


_SPATIAL = ("locations", "slots", "links", "merchants")
_STAGE_LIMIT = 64 * 1024**2


def _layout(tensor):
    try:
        version = tensor._version
    except RuntimeError:  # inference-mode tensors have no version counter
        return None
    return (version, tensor.device, tensor.dtype, tuple(tensor.shape),
            tensor.stride(), tensor.data_ptr())


@dataclass(frozen=True)
class _StagedPublic(EncodedPublic):
    """Only the private packer binds exact host bytes to the new device views."""
    _spatial_bytes: tuple
    _source_tensors: tuple
    _source_layouts: tuple

    def spatial_key(self):
        for name, source, layout in zip(_SPATIAL, self._source_tensors, self._source_layouts):
            current = getattr(self, name)
            if layout is None or current is not source or _layout(current) != layout:
                return None
        return self._spatial_bytes


def staged_spatial_key(public):
    if type(public) is _StagedPublic:
        return public.spatial_key()
    return None


def _groups(records, record_type, single_move=False):
    groups = {}
    for field in fields(record_type):
        arrays = [getattr(record, field.name) for record in records]
        first = arrays[0]
        if not isinstance(first, np.ndarray) or not first.size:
            return None
        stride = first.itemsize
        canonical = []
        for size in reversed(first.shape):
            canonical.append(stride)
            stride *= size
        strides = tuple(reversed(canonical))
        if any(not isinstance(a, np.ndarray) or a.ndim == 0 or a.strides != strides
               or a.dtype.hasobject or a.dtype != first.dtype or a.shape != first.shape
               or (single_move and len(a) != 1) for a in arrays):
            return None
        groups[field.name] = arrays
    return groups


def _stage(events, device, remaining):
    view_count = 2 * len(fields(EncodedPublic)) + len(fields(EncodedCandidates))
    minimum = len(events) * (512 * view_count + 4096)
    if not events or minimum > remaining:
        return events, remaining
    boards = [e.state for e in events] + [e.delta for e in events]
    public = _groups(boards, EncodedPublic)
    moves = _groups([e.move for e in events], EncodedCandidates, single_move=True)
    if public is None or moves is None:
        return events, remaining
    groups = list(public.values()) + list(moves.values())
    sizes = [len(group) * group[0].nbytes for group in groups]
    # Charge rounded device allocations, saved spatial bytes, tensor views,
    # Python bookkeeping and the largest simultaneously live host stack.
    retained = sum((size + 511) // 512 * 512 for size in sizes)
    retained += sum(len(public[k]) * public[k][0].nbytes for k in _SPATIAL)
    retained += 512 * sum(map(len, groups)) + 2048 * len(boards)
    if retained + max(sizes) > remaining:
        return events, remaining

    columns = {}
    signatures = [[] for _ in boards]
    for field, arrays in public.items():
        host = np.stack(arrays)
        columns[field] = torch.as_tensor(host, device=device).unbind(0)
        if field in _SPATIAL:
            for i, row in enumerate(host):
                signatures[i].append((row.dtype.str, row.shape, row.tobytes()))
            del row
        del host

    staged_boards = []
    for i, signature in enumerate(signatures):
        values = {name: chunks[i] for name, chunks in columns.items()}
        sources = tuple(values[k] for k in _SPATIAL)
        staged_boards.append(_StagedPublic(**values, _spatial_bytes=tuple(signature),
                             _source_tensors=sources, _source_layouts=tuple(map(_layout, sources))))
    move_columns = {name: torch.as_tensor(np.concatenate(arrays), device=device).split(1)
                    for name, arrays in moves.items()}
    placed = tuple(replace(e, state=staged_boards[i], delta=staged_boards[len(events)+i],
                           move=EncodedCandidates(**{k: chunks[i] for k, chunks in move_columns.items()}))
                   for i, e in enumerate(events))
    return placed, remaining - retained


def place_events(batches, device, limit=_STAGE_LIMIT):
    """Return the complete original batches if optional placement runs out of memory."""
    remaining = min(limit, _STAGE_LIMIT)
    if device.type == "cuda":
        ordinal = device.index if device.index is not None else torch.cuda.current_device()
        free, _ = torch.cuda.mem_get_info(ordinal)
        remaining = min(remaining, free // 16)
    # The input is the private update-local list. Delay all assignments until
    # successful packing, so even OOM after earlier batches leaves it intact.
    bookkeeping = 1024
    replacements = []
    try:
        for index, (batch, prepared) in enumerate(batches):
            if remaining <= bookkeeping:
                break
            allowance = remaining - bookkeeping
            burn, allowance = _stage(prepared.burn_in_events, device, allowance)
            main, allowance = _stage(prepared.main_events, device, allowance)
            if burn is not prepared.burn_in_events or main is not prepared.main_events:
                updated = replace(prepared, burn_in_events=burn, main_events=main)
                replacements.append((index, (batch, updated)))
                remaining = allowance
    except (MemoryError, torch.cuda.OutOfMemoryError):
        return batches
    for index, replacement in replacements:
        batches[index] = replacement
    return batches
