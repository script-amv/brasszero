# Model card for {{MODEL}}

## Model details

| Field | Value |
| --- | --- |
| Portable policy | {{POLICY}} |
| Update | {{UPDATE}} |
| Lineage and branch | {{LINEAGE}}, {{BRANCH}} |
| Parameters | {{PARAMETERS}} |
| Configuration | {{CONFIG}} |
| Engine | {{ENGINE}} |
| Schema | {{SCHEMA}} |
| SHA-256 of policy.pt | {{WEIGHTS}} |
| Licence | Apache-2.0 |

## Use and training

This policy plays four-player Brass: Birmingham from native observations and public events. It was trained with self-play and PPO. Intended uses include gameplay, evaluation, and research. The release reviewer should fill the parent checkpoint, training dates, number of training games and opponent mixture from the experiment journal.

## Evaluation and limitations

{{EVALUATION}} Report strict wins after all native tiebreakers, the number of independent deals and seat rotations, and paired intervals. Training replicates are limited. Playing strength against expert human opponents remains unmeasured.
