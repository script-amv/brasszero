"""Functional inference: native observations/public events, never Game objects."""
from dataclasses import fields, replace
import hashlib
import json
import math

import torch

from brass_rl.distributions import PolicyOutput, policy_probabilities
from brass_rl.features import EncodedCandidates, encode_observation, encode_public
from .allocation import BudgetStop


def move_key(move_json):
    return json.dumps(json.loads(move_json), sort_keys=True, separators=(',', ':'), allow_nan=False)


def domain_seed(seed, seat, ply, domain, index=0):
    data = json.dumps([seed, seat, ply, domain, index], separators=(',', ':')).encode()
    return int.from_bytes(hashlib.blake2b(data, digest_size=8, person=b'brass-search-v1').digest(), 'little')


def standard_gumbel(seed, key):
    digest = hashlib.blake2b(seed.to_bytes(8, 'little') + key.encode(), digest_size=8,
                             person=b'brass-root-gumb').digest()
    # 53 significant random bits, strictly inside (0,1).
    u = ((int.from_bytes(digest, 'little') >> 11) + 1) / (2**53 + 2)
    return -math.log(-math.log(u))


def check_wall(deadline):
    import time
    if deadline is not None and time.monotonic() >= deadline:
        raise BudgetStop('wall_cap')


def forward(model, components, obs, memory, config, counters, *, encoded=None, root=False, canonical=True):
    if root:
        canonical = False  # Mandatory exact TorchAgent baseline sampler input.
    if encoded is None:
        encoded = encode_observation(obs, components)
    keys = [move_key(move) for move in obs.legal_moves()]
    order = sorted(range(len(keys)), key=lambda i: keys[i]) if canonical else list(range(len(keys)))
    if canonical:
        candidates = EncodedCandidates(**{field.name: getattr(encoded.candidates, field.name)[order]
                                          for field in fields(encoded.candidates)})
        encoded = replace(encoded, candidates=candidates)
    counters['root_candidates_encoded' if root else 'rollout_candidates_encoded'] += len(keys)
    counters['inference_calls'] += 1
    counters['policy_forward_calls'] += 1
    with torch.no_grad():
        output = model([encoded], memory, candidate_chunk=1024 if root else config.candidate_chunk)
    probabilities = policy_probabilities(output)  # validates logits and complete row shapes
    if (len(output.values) != 1 or len(probabilities) != len(keys) or
            not output.valid.all().item() or not torch.isfinite(output.values).all().item()):
        raise ValueError('policy must return complete legal scores and finite values')
    # Return native order to preserve live baseline sampling semantics.
    inverse = sorted(range(len(order)), key=lambda i: order[i])
    return PolicyOutput(output.logits[inverse], output.offsets, output.valid[inverse], output.values)


def advance_memories(model, components, memories, event, previous, counters=None, deadline=None, skip_viewer=None):
    updated = []
    for viewer in range(4):
        if viewer == skip_viewer:
            updated.append(memories[viewer])
            continue
        check_wall(deadline)
        encoded = encode_public(event, viewer, previous, components)
        check_wall(deadline)
        with torch.no_grad():
            memory = model.update_memory([encoded], memories[viewer]).detach()
        if not torch.isfinite(memory).all().item():
            raise ValueError('public memory must be finite')
        if counters is not None:
            counters['inference_calls'] += 1
            counters['memory_update_calls'] += 1
        updated.append(memory)
    return updated
