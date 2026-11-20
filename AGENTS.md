# Engineering conventions

Read the usage, results and maintenance guides before changing a public interface. Run `make test lint check-public`. Use `make test-training` when modifying optional training code.

Only `GameState::apply` mutates a game. A player receives its own observation and sanitized public events. Hidden state or recoverable deal seeds must stay outside player code. Report sole first place after native tiebreakers and disclose the number of independent deals.

Treat model weights, update envelopes and replay records as immutable. Verify checksums at transport boundaries. Changing a source-identity input creates a new compatibility identity. Keep exact source bytes for preserved formats unless you implement and test a versioned migration. Keep private reference engines, GUI, third-party game material and local writer state outside public history. No automatic publishing or training continuation.
