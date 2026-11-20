# SPDX-License-Identifier: Apache-2.0
"""Original scalar spatial graph round with a device-specific execution choice."""
import sys
from typing import Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


def eager_graph_round(loc: Tensor, slots: Tensor, links: Tensor, merchants: Tensor,
                       slot_inc: Tensor, merchant_inc: Tensor, link_inc: Tensor,
                       slot_locs: Tensor, merchant_locs: Tensor,
                       lw: Tensor, lb: Tensor, ew: Tensor, eb: Tensor,
                       sw: Tensor, sb: Tensor, mw: Tensor, mb: Tensor) -> Tuple[Tensor,Tensor,Tensor,Tensor]:
    local = slot_inc.T @ slots + merchant_inc.T @ merchants
    local = local / (slot_inc.sum(0) + merchant_inc.sum(0)).clamp_min(1)[:, None]
    edges = link_inc.T @ links / link_inc.sum(0).clamp_min(1)[:, None]
    next_loc = torch.tanh(F.linear(torch.cat((loc, local, edges), dim=-1), lw, lb))
    ends = link_inc @ loc / link_inc.sum(1)[:, None]
    links = torch.tanh(F.linear(torch.cat((links, ends), dim=-1),ew,eb))
    slots = torch.tanh(F.linear(torch.cat((slots, loc[slot_locs]), dim=-1),sw,sb))
    merchants = torch.tanh(F.linear(torch.cat((merchants, loc[merchant_locs]), dim=-1),mw,mb))
    return next_loc, slots, links, merchants


_cpu_graph_round = torch.jit.script(eager_graph_round)


def graph_round_for(device: torch.device):
    # On the measured CUDA backend TorchScript changed a small PPO gradient
    # enough to move Adam parameters outside the registered error bound.
    # On the pinned macOS CPU runtime the warmed scripted autograd graph
    # mis-broadcasts the 27 location rows across the 64 embedding channels.
    if device.type == "cpu" and not (sys.platform == "darwin" and torch.is_grad_enabled()):
        return _cpu_graph_round
    return eager_graph_round
