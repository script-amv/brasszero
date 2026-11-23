# SPDX-License-Identifier: Apache-2.0
"""Versioned dual-Adam payload stored at the existing sparse optimizer boundary."""
from copy import deepcopy
import math

import torch

from .policy_io import ArtifactError


FORMAT = 'brass-stability-optimizers/1'


def _validate_adam(model, state, lr):
    if not isinstance(state, dict) or set(state) != {'state', 'param_groups'} or len(state['param_groups']) != 1:
        raise ArtifactError('invalid stability Adam envelope')
    group = state['param_groups'][0]
    parameters = list(model.parameters())
    if group['params'] != list(range(len(parameters))) or group['lr'] != lr:
        raise ArtifactError('stability Adam must preserve parameter order and learning rate')
    if (len(group['betas']) != 2 or not all(0 <= v < 1 for v in group['betas'])
            or not math.isfinite(group['eps']) or group['eps'] < 0
            or not math.isfinite(group['weight_decay']) or group['weight_decay'] < 0):
        raise ArtifactError('invalid stability Adam hyperparameters')
    moments = {'exp_avg', 'exp_avg_sq'}
    if group['amsgrad']:
        moments.add('max_exp_avg_sq')
    for index, slot in state['state'].items():
        if type(index) is not int or index not in range(len(parameters)) or set(slot) != moments | {'step'}:
            raise ArtifactError('invalid stability Adam parameter slot')
        step = slot['step']
        if (not isinstance(step, torch.Tensor) or step.ndim != 0 or not torch.isfinite(step).item()
                or step.item() < 0 or not float(step.item()).is_integer()):
            raise ArtifactError('invalid stability Adam step counter')
        for key in moments:
            tensor = slot[key]
            param = parameters[index]
            if (not isinstance(tensor, torch.Tensor) or tensor.shape != param.shape or tensor.dtype != param.dtype
                    or not torch.isfinite(tensor).all().item()
                    or (key != 'exp_avg' and (tensor < 0).any().item())):
                raise ArtifactError('invalid stability Adam moment: '+key)


def unpack(actor, payload, lr):
    if (not isinstance(payload, dict) or set(payload) != {'format', 'actor_adam', 'critic_weights', 'critic_adam'}
            or payload['format'] != FORMAT):
        raise ArtifactError('candidate requires a complete versioned stability optimizer checkpoint')
    weights = payload['critic_weights']
    expected = actor.state_dict()
    if not isinstance(weights, dict) or weights.keys() != expected.keys():
        raise ArtifactError('invalid independent critic keys')
    for name, value in expected.items():
        saved = weights[name]
        if (not isinstance(saved, torch.Tensor) or saved.shape != value.shape or saved.dtype != value.dtype
                or not torch.isfinite(saved).all().item()):
            raise ArtifactError('invalid independent critic tensor: '+name)
    if any(not torch.equal(weights[name].cpu(), value.cpu()) for name, value in actor.named_buffers()):
        raise ArtifactError('independent critic static board mismatch')
    _validate_adam(actor, payload['actor_adam'], lr)
    _validate_adam(actor, payload['critic_adam'], lr)
    return payload['actor_adam'], weights, payload['critic_adam']


def pack(actor, actor_adam, critic, critic_adam, lr):
    if critic is None or not isinstance(critic_adam, torch.optim.Adam) or critic.config != actor.config:
        raise ArtifactError('candidate checkpoint requires a matching independent critic and Adam')
    if ([id(p) for p in critic.parameters()] != [id(p) for g in critic_adam.param_groups for p in g['params']]
            or {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}):
        raise ArtifactError('independent critic must own its parameters and Adam in canonical order')
    payload = {'format': FORMAT, 'actor_adam': actor_adam.state_dict(),
               'critic_weights': critic.state_dict(), 'critic_adam': critic_adam.state_dict()}
    unpack(actor, payload, lr)
    return payload


def independent_critic(loaded, *, clone=False):
    critic = deepcopy(loaded.model)
    adam = torch.optim.Adam(critic.parameters(), lr=loaded.state.config.ppo.learning_rate)
    if clone:
        if loaded.critic_state is not None or loaded.critic_optimizer_state is not None:
            raise ValueError('refusing to reset independent critic or Adam')
    else:
        critic.load_state_dict(loaded.critic_state)
        adam.load_state_dict(loaded.critic_optimizer_state)
    return critic, adam
