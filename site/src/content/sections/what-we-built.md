---
order: 2
kicker: What we built
title: From the rulebook to self-play
figure:
  image: ../../assets/art/arkwright.jpg
  alt: Arkwright's Mills by Joseph Wright of Derby. A tall cotton mill stands in a wooded river gorge.
  caption: Joseph Wright of Derby, Arkwright's Mills, about 1795.
---
1. **A verified rules engine.** Written in Rust from the official rulebook and checked move by move against two independent implementations. It plays 530–555 complete games per second on a single laptop core, fast enough for self-play without a cluster.
2. **A graph neural network.** 390,342 parameters that read the board as connected cities, links and players. For every legal move it estimates how good the move is, and for every player how likely they are to finish first.
3. **Self-play training.** The model plays against a league of its own earlier versions (PPO). Each checkpoint keeps its exact identity and lineage.
4. **Evidence you can audit.** Models are compared on the same deals with every seat rotation, and results come with 95% intervals. Games are recorded automatically and can be replayed move by move.
