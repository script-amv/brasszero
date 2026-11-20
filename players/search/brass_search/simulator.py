"""Hypothetical games created exclusively by Observation.determinize.

One sampled starting world and one cloned continuation at most. No recorder, true
state, referee seed, neural training state, or per-step tensor history is accepted.
"""
import json

import torch

from brass_ai.events import PublicEvent
from brass_rl.distributions import policy_probabilities
from .allocation import relative_to_absolute, terminal_values
from .inference import advance_memories, check_wall, domain_seed, forward, move_key


def normalized_observation(obs):
    view = json.loads(obs.to_json())
    view['my_hand'] = sorted(view['my_hand'], key=lambda x: json.dumps(x, sort_keys=True))
    view['legal_moves'] = sorted(view['legal_moves'], key=lambda x: json.dumps(x, sort_keys=True))
    return view


class SampledSimulator:
    def __init__(self, model, components, obs, memories, seed, ply, config, counters, deadline):
        self.model, self.components, self.obs = model, components, obs
        self.memories, self.seed, self.ply = memories, seed, ply
        self.config, self.counters, self.deadline = config, counters, deadline
        self._world, self._world_index = None, None
        self._root_view = normalized_observation(obs)

    def __call__(self, root_move, world_index):
        check_wall(self.deadline)
        if world_index != self._world_index:
            self._world = None
            self._world = self.obs.determinize(domain_seed(self.seed, self.obs.me, self.ply, 'world', world_index))
            # Creation already spent work and owns a live world even if the next
            # cooperative boundary stops before validation or continuation cloning.
            self.counters['worlds_sampled'] += 1
            self.counters['max_live_worlds'] = max(self.counters['max_live_worlds'], 1)
            check_wall(self.deadline)
            if normalized_observation(self._world.observation(self.obs.me)) != self._root_view:
                raise ValueError('sampled root observation or legal moves mismatch')
            self._world_index = world_index
        check_wall(self.deadline)
        game = self._world.clone()
        self.counters['max_live_worlds'] = max(self.counters['max_live_worlds'], 2)
        self.counters['max_live_continuations'] = 1
        memories = [memory.detach().clone() for memory in self.memories]
        generator = torch.Generator(device='cpu')
        generator.manual_seed(domain_seed(self.seed, self.obs.me, self.ply, 'rollout', world_index))
        previous = self.obs.public_json()
        move = root_move
        for depth in range(self.config.rollout_depth):
            check_wall(self.deadline)
            actor = game.current_player()
            if depth:
                obs = game.observation(actor)
                check_wall(self.deadline)
                output = forward(self.model, self.components, obs, memories[actor], self.config, self.counters)
                check_wall(self.deadline)
                keys = [move_key(m) for m in obs.legal_moves()]
                order = sorted(range(len(keys)), key=lambda i: keys[i])
                p = policy_probabilities(output).detach().cpu()[order]
                move = keys[order[int(torch.multinomial(p, 1, generator=generator).item())]]
            check_wall(self.deadline)
            game.apply(move)  # Native validated mutation, including forced root and follow-ups.
            self.counters['native_applies'] += 1
            check_wall(self.deadline)
            public = game.observation(actor).public_json()
            event = PublicEvent(self.ply + depth + 1, actor, move, public)
            memories = advance_memories(self.model, self.components, memories, event, previous,
                                       self.counters, self.deadline)
            previous = public
            check_wall(self.deadline)
            terminal = game.is_over()
            check_wall(self.deadline)
            if terminal:
                value = terminal_values(game.ranking())[self.obs.me]
                check_wall(self.deadline)
                return value
        check_wall(self.deadline)
        actor = game.current_player()
        obs = game.observation(actor)
        check_wall(self.deadline)
        output = forward(self.model, self.components, obs, memories[actor], self.config, self.counters)
        check_wall(self.deadline)
        return relative_to_absolute(output.values[0].detach().cpu().tolist(), actor)[self.obs.me]
