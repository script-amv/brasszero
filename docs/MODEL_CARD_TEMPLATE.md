# Model card for {{MODEL}}

## Model details

| Field | Value |
| --- | --- |
| Portable policy | {{POLICY}} |
| Update | {{UPDATE}} |
| Lineage and branch | {{LINEAGE}}, {{BRANCH}} |
| Parent checkpoint | {{PARENT}} |
| Experiment started | {{DATE}} |
| Parameters | {{PARAMETERS}} |
| Configuration | {{CONFIG}} |
| Engine | {{ENGINE}} |
| Schema | {{SCHEMA}} |
| SHA-256 of policy.pt | {{WEIGHTS}} |
| Licence | Apache-2.0 |

## Use and training

This policy plays four-player Brass: Birmingham from native observations and public events. It was trained with PPO against itself and a league of opponents. Intended uses include gameplay, evaluation, and research. {{TRAINING}} The [model repository](https://huggingface.co/brasszero/brasszero-ppo) describes the full training line.

## Evaluation and limitations

{{EVALUATION}}

Sole wins use all native tiebreakers. Training replicates are limited. Playing strength against expert human opponents remains unmeasured.
