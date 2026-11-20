# Replay archives

Local archives contain compressed completed games under `games/`, transient JSONL journals, advisory files under `locks/`, and a rebuildable `index.jsonl`. Each index line is one summary. The public game dataset uses the versioned projection in the datasheet and never includes the local index. Generate it with `make dataset-snapshot`. See [the datasheet](../docs/DATASHEET.md).
