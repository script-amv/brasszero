# SPDX-License-Identifier: Apache-2.0
"""Recurrent PPO with batch-wide old-policy validation and transactional aborts.

Each sequence starts from a stale, detached *rollout* boundary h[s], burns in
under current weights without gradients, then recomputes learned event/graph
embeddings and GRU states with gradients inside its main interval. This is the
explicit truncated-BPTT approximation. Model-independent input encodings may be
reused across the update. No-grad spatial embeddings may be reused only within
one fixed-weight sequence invocation; the differentiable graph is rebuilt.
"""
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from typing import Mapping, Tuple

import numpy as np
import torch
from torch import Tensor

from .board_cache import fixed_boards
from .device_events import place_events
from .config import PPOConfig
from .distributions import PolicyStats, policy_stats
from .features import EncodedEvent, EncodedObservation, encode_observation, encode_public
from .returns import episode_targets
from .schema import ComponentData
from .sequences import SequenceBatch, sequence_batches


@dataclass(frozen=True)
class LossTerms:
    policy: Tensor
    value: Tensor
    entropy: Tensor
    total: Tensor
    approx_kl: Tensor
    clip_fraction: Tensor


@dataclass(frozen=True)
class UpdateMetrics:
    policy: float = 0.
    value: float = 0.
    entropy: float = 0.
    total: float = 0.
    approx_kl: float = 0.
    clip_fraction: float = 0.
    calibration: float = 0.
    gradient_norm: float = 0.
    optimizer_steps: int = 0
    kl_stopped: bool = False
    max_reconstructed_logp_error: float = 0.
    policy_decisions: int = 0
    value_decisions: int = 0


def _finite(value, name):
    if isinstance(value, Tensor):
        valid = torch.isfinite(value).all().item()
    else:
        valid = np.isfinite(value).all()
    if not valid:
        raise ValueError('%s must be finite; PPO update cannot continue' % name)


def _validate_loss_terms(terms: LossTerms) -> None:
    # Finite inputs/ratios do not guarantee finite reductions (notably KL).
    for field in fields(terms):
        _finite(getattr(terms, field.name), field.name + ' loss/diagnostic')


def normalize_targets(targets: Mapping) -> dict:
    """Normalize raw advantages once across ALL eligible rollout decisions.

    Population variance is used; empty and zero-variance sets produce zeros.
    Ineligible singleton advantages become zero and never affect statistics.
    """
    eligible = []
    for target in targets.values():
        _finite(target.self_advantage, 'raw advantage')
        _finite(target.value_targets, 'value targets')
        if target.policy_mask:
            eligible.append(target.self_advantage)
    mean = float(np.mean(eligible, dtype=np.float64)) if eligible else 0.
    std = float(np.std(eligible, dtype=np.float64)) if eligible else 0.
    return {key: replace(target, self_advantage=(
        (target.self_advantage-mean)/std if target.policy_mask and std > 0. else 0.))
        for key, target in targets.items()}


def ppo_loss(new: PolicyStats, old_logp, advantages, value_targets, policy_mask,
             config: PPOConfig) -> LossTerms:
    """Exact complete-list PPO; singleton rows contribute only four-value MSE."""
    device, dtype = new.log_prob.device, new.log_prob.dtype
    old = torch.as_tensor(old_logp, device=device, dtype=dtype)
    advantages = torch.as_tensor(advantages, device=device, dtype=dtype)
    targets = torch.as_tensor(value_targets, device=device, dtype=new.values.dtype)
    mask = torch.as_tensor(policy_mask, device=device)
    n = len(new.log_prob)
    if (n == 0 or new.log_prob.shape != (n,) or new.entropy.shape != (n,)
            or new.values.shape != (n, 4) or old.shape != (n,) or advantages.shape != (n,)
            or targets.shape != (n, 4) or mask.shape != (n,) or mask.dtype != torch.bool):
        raise ValueError('PPO needs logp/entropy/old/advantage/mask [D] and values/targets [D,4]')
    for name, data in (('new log probabilities', new.log_prob), ('entropy', new.entropy),
                       ('values', new.values), ('old log probabilities', old),
                       ('advantages', advantages), ('value targets', targets)):
        _finite(data, name)
    # expm1(log_ratio)-log_ratio is (ratio-1)-log(ratio), stable near zero.
    log_ratio = new.log_prob[mask]-old[mask]
    ratio = log_ratio.exp()
    _finite(ratio, 'probability ratios')
    if mask.any().item():
        surrogate = torch.minimum(ratio*advantages[mask],
            ratio.clamp(1-config.clip_epsilon, 1+config.clip_epsilon)*advantages[mask])
        policy = -surrogate.mean()
        entropy = new.entropy[mask].mean()
        kl = (log_ratio.expm1()-log_ratio).mean()
        clipped = ((ratio-1).abs() > config.clip_epsilon).to(dtype).mean()
    else:
        policy = new.log_prob.sum()*0.
        entropy = new.entropy.sum()*0.
        kl = clipped = policy
    value = (new.values-targets).square().mean()
    total = policy+config.value_coef*value-config.entropy_coef*entropy
    terms = LossTerms(policy, value, entropy, total, kl, clipped)
    _validate_loss_terms(terms)
    return terms


@dataclass(frozen=True)
class _PreparedSequence:
    """Update-local raw arrays only; no learned embeddings or memory states."""
    burn_in_events: Tuple[EncodedEvent, ...]
    main_events: Tuple[EncodedEvent, ...]
    observations: Tuple[EncodedObservation, ...]


def _prepare_sequence(batch: SequenceBatch, components) -> _PreparedSequence:
    previous = batch.previous_public_json
    events = []
    for event in (*batch.burn_in_events, *batch.main_events):
        events.append(encode_public(event, batch.viewer, previous, components))
        previous = event.public_json
    burn = len(batch.burn_in_events)
    return _PreparedSequence(tuple(events[:burn]), tuple(events[burn:]),
        tuple(encode_observation(d.observation, components) for d in batch.decisions))


@fixed_boards()
def _sequence_stats(model, batch: SequenceBatch, components, prepared=None) -> PolicyStats:
    memory = batch.recorded_start_state.to(device=model.device).detach()
    previous = batch.previous_public_json
    with torch.no_grad():
        for index, event in enumerate(batch.burn_in_events):
            encoded = (encode_public(event, batch.viewer, previous, components) if prepared is None
                       else prepared.burn_in_events[index])
            memory = model.update_memory([encoded], memory)
            previous = event.public_json
    memory = memory.detach()
    memories, observations, choices = [], [], []
    decisions = {d.ply: d for d in batch.decisions}
    for k, event in enumerate(batch.main_events, batch.start_ply):
        # Decision k is before event k+1. Retain the graph through earlier events.
        if k in decisions:
            decision = decisions[k]
            memories.append(memory)
            observations.append(encode_observation(decision.observation, components) if prepared is None
                                else prepared.observations[len(observations)])
            choices.append(decision.index)
        encoded = (encode_public(event, batch.viewer, previous, components) if prepared is None
                   else prepared.main_events[k-batch.start_ply])
        memory = model.update_memory([encoded], memory)
        previous = event.public_json
    output = model(observations, torch.cat(memories, dim=0))
    return policy_stats(output, torch.tensor(choices, dtype=torch.int64, device=model.device))


def _loss(model, batch, components, config, prepared=None):
    new = _sequence_stats(model, batch, components, prepared)
    return new, ppo_loss(new, [d.old_log_prob for d in batch.decisions],
        [t.self_advantage for t in batch.targets], np.stack([t.value_targets for t in batch.targets]),
        [t.policy_mask for t in batch.targets], config)


def _check_state(model, optimizer):
    for name, value in model.state_dict().items():
        _finite(value, 'model state '+name)
    def check(value):
        if isinstance(value, Tensor):
            _finite(value, 'optimizer state')
        elif isinstance(value, dict):
            for child in value.values():
                check(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                check(child)
        elif isinstance(value, float):
            _finite(value, 'optimizer state')
    check(optimizer.state_dict())


def _rng_snapshot(rng):
    if hasattr(rng, 'getstate'):
        return ('python', deepcopy(rng.getstate()))
    if hasattr(rng, 'bit_generator'):
        return ('numpy', deepcopy(rng.bit_generator.state))
    if hasattr(rng, 'get_state'):
        return ('numpy_legacy', deepcopy(rng.get_state()))
    raise ValueError('minibatch rng must support shuffle and restorable state')


def _rng_restore(rng, state):
    kind, value = state
    if kind == 'python':
        rng.setstate(value)
    elif kind == 'numpy':
        rng.bit_generator.state = value
    else:
        rng.set_state(value)


def update_policy(model, optimizer, episodes, config: PPOConfig, rng) -> UpdateMetrics:
    """Update current weights; optimizer belongs to the caller, never rollouts.

    Every recorded logp (singletons included) is checked before the first step.
    KL is re-evaluated after each step on that sequence minibatch. A finite KL
    early stop preserves completed steps. Any exception restores the pre-update
    model, optimizer and consumed minibatch RNG state, clears gradients and raises.

    Metrics average policy/entropy/KL/clip by eligible decision-visits and value
    MSE/calibration by all current decision-visits across completed steps. Losses
    use pre-step stats; KL uses post-step stats. Gradient norm is mean pre-clip
    norm. Counts report unique rollout decisions, not epoch-expanded visits.
    Calibration is mean squared error of mean predictions vs mean targets, per
    component, aggregated over completed minibatches (targets are GAE returns).
    """
    saved_model, saved_optimizer = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
    saved_rng, training = _rng_snapshot(rng), model.training
    steps, policy_visits, value_visits, stopped = 0, 0, 0, False
    sums = dict(policy=0., value=0., entropy=0., approx_kl=0., clip_fraction=0., calibration=0., gradient_norm=0.)
    max_error = 0.
    try:
        _check_state(model, optimizer)
        model.eval()  # deterministic rollout reconstruction; autograd stays enabled for updates
        components = ComponentData.load()
        if components.schema.fingerprint != model.schema.fingerprint:
            raise ValueError('PPO model/component schema mismatch')
        targets = {}
        for episode in episodes:
            target = episode_targets(episode, gamma=config.gamma, lam=config.gae_lambda,
                                     reward=config.reward, critic_target=config.critic_target)
            if targets.keys() & target.keys():
                raise ValueError('duplicate rollout decision keys')
            targets.update(target)
        targets = normalize_targets(targets)
        batches = [(batch, _prepare_sequence(batch, components)) for batch in sequence_batches(
            episodes, targets, config.sequence_events, config.burn_in_events, rng)]
        if model.device.type == 'cuda':
            batches = place_events(batches, model.device)
        # The complete batch must validate before ANY optimizer step; no lazy
        # per-minibatch validation that could accept an already-corrupted update.
        with torch.no_grad():
            for batch, prepared in batches:
                new = _sequence_stats(model, batch, components, prepared)
                old = new.log_prob.new_tensor([d.old_log_prob for d in batch.decisions])
                _finite(old, 'old rollout log probabilities')
                _finite(new.log_prob, 'reconstructed log probabilities')
                _finite(new.entropy, 'reconstructed entropy')
                _finite(new.values, 'reconstructed values')
                error = float((new.log_prob-old).abs().max().item())
                max_error = max(max_error, error)
                if error > 1e-5:
                    raise ValueError('rollout logp reconstruction error %.9g > 1e-5 (game %s seat %s)' %
                                     (error, batch.game_id, batch.viewer))
        for epoch in range(config.epochs):
            if epoch:
                rng.shuffle(batches)
            for batch, prepared in batches:
                optimizer.zero_grad(set_to_none=True)
                new, terms = _loss(model, batch, components, config, prepared)
                _validate_loss_terms(terms)
                terms.total.backward()
                for parameter in model.parameters():
                    if parameter.grad is not None:
                        _finite(parameter.grad, 'gradients')
                gradient_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm,
                                                              error_if_nonfinite=True)
                _finite(gradient_norm, 'gradient norm')
                optimizer.step()
                _check_state(model, optimizer)
                with torch.no_grad():
                    _, post = _loss(model, batch, components, config, prepared)
                _validate_loss_terms(post)
                eligible = sum(t.policy_mask for t in batch.targets)
                count = len(batch.decisions)
                for name in ('policy', 'entropy', 'clip_fraction'):
                    sums[name] += float(getattr(terms, name).detach().item())*eligible
                sums['approx_kl'] += post.approx_kl.item()*eligible
                sums['value'] += terms.value.detach().item()*count
                target_values = new.values.new_tensor(np.stack([t.value_targets for t in batch.targets]))
                calibration = (new.values.detach().mean(0)-target_values.mean(0)).square().mean().item()
                sums['calibration'] += calibration*count
                sums['gradient_norm'] += float(gradient_norm.item())
                policy_visits += eligible
                value_visits += count
                steps += 1
                if post.approx_kl.item() > config.target_kl:
                    stopped = True
                    break
            if stopped:
                break
        metrics = {name: sums[name]/max(1, policy_visits) for name in ('policy','entropy','approx_kl','clip_fraction')}
        metrics.update({name: sums[name]/max(1, value_visits) for name in ('value','calibration')})
        metrics['gradient_norm'] = sums['gradient_norm']/max(1, steps)
        metrics['total'] = metrics['policy']+config.value_coef*metrics['value']-config.entropy_coef*metrics['entropy']
        optimizer.zero_grad(set_to_none=True)
        return UpdateMetrics(**metrics, optimizer_steps=steps, kl_stopped=stopped,
                             max_reconstructed_logp_error=max_error, value_decisions=len(targets),
                             policy_decisions=sum(t.policy_mask for t in targets.values()))
    except Exception as error:
        model.load_state_dict(saved_model)
        optimizer.load_state_dict(saved_optimizer)
        _rng_restore(rng, saved_rng)
        optimizer.zero_grad(set_to_none=True)
        raise ValueError('PPO update aborted and rolled back: %s' % error) from error
    finally:
        model.train(training)
