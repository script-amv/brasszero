"""Ragged complete-move distributions with one exact normalizer per decision."""
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass(frozen=True)
class PolicyOutput:
    logits: Tensor
    offsets: Tensor
    valid: Tensor
    values: Tensor


@dataclass(frozen=True)
class PolicyStats:
    log_prob: Tensor
    entropy: Tensor
    values: Tensor


def _rows(output: PolicyOutput) -> List[Tuple[Tensor, Tensor, Tensor]]:
    logits, offsets, valid, values = output.logits, output.offsets, output.valid, output.values
    if (logits.ndim != 1 or offsets.ndim != 1 or offsets.dtype != torch.int64
            or valid.shape != logits.shape or valid.dtype != torch.bool
            or values.ndim != 2 or values.shape[1] != 4
            or len(offsets) != len(values) + 1 or len(values) == 0):
        raise ValueError("invalid policy shapes/dtypes")
    boundaries = offsets.detach().cpu().tolist()
    if boundaries[0] != 0 or boundaries[-1] != len(logits):
        raise ValueError("policy offsets must cover all candidates")
    rows = []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        if end <= start or not valid[start:end].any().item():
            raise ValueError("decision needs at least one valid candidate")
        mask = valid[start:end]
        row = logits[start:end]
        if not torch.isfinite(row[mask]).all().item():
            raise ValueError("valid candidate logits must be finite")
        logp = torch.log_softmax(row.masked_fill(~mask, -torch.inf), dim=0)
        # Zeroing masked log-probabilities also avoids 0 * -inf in entropy.
        logp = logp.masked_fill(~mask, 0.)
        probability = logp.exp().masked_fill(~mask, 0.)
        rows.append((logp, probability, mask))
    return rows


def policy_probabilities(output: PolicyOutput) -> Tensor:
    """Flat probabilities in original candidate order; padding has exactly zero mass."""
    return torch.cat([prob for _, prob, _ in _rows(output)])


def policy_stats(output: PolicyOutput, choices: Tensor) -> PolicyStats:
    """Choices are local legal-list indices, never flat offsets."""
    rows = _rows(output)
    if choices.dtype != torch.int64 or choices.shape != (len(rows),):
        raise ValueError("choices must be one int64 local index per decision")
    logs, entropies = [], []
    for (logp, prob, valid), choice in zip(rows, choices.detach().cpu().tolist()):
        if not 0 <= choice < len(logp) or not valid[choice].item():
            raise ValueError("choice is not a valid local candidate index")
        logs.append(logp[choice])
        entropies.append(-(prob * logp).sum())
    return PolicyStats(torch.stack(logs), torch.stack(entropies), output.values)
