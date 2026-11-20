SHELL := /bin/sh
PYTHONPATH := $(CURDIR)/engine/tools:$(CURDIR)/training:$(CURDIR)/players/search:$(CURDIR)/players/llm
export PYTHONPATH
export OMP_NUM_THREADS := 1
export MKL_NUM_THREADS := 1
export OPENBLAS_NUM_THREADS := 1
export RUST_TEST_THREADS := 1
export CARGO_BUILD_JOBS ?= 1
SITE_PORT ?= 4321
GUI_PORT ?= 8773
CONFIRM_FLAG = $(if $(filter 1,$(CONFIRM)),--confirm,)
.PHONY: setup setup-training test test-rust test-py test-training test-site lint lint-python check-public play site-dev site-data publish-progress release-model dataset-snapshot release sync-check
setup:
	@echo "Install core Python, native bindings, site dependencies and local hooks"
	@sh scripts/setup.sh
setup-training:
	@echo "Install the optional verified CPU training environment"
	@sh scripts/setup-training.sh
test: test-rust test-py test-site
test-rust:
	@echo "Test the Rust workspace with the locked dependency graph"
	@cargo test --workspace --locked
test-py:
	@echo "Test core native bindings, referee, public players and standard-library tooling"
	@.venv/bin/python -m pytest -q engine/bindings/tests engine/tools/tests players/llm/tests players/search/tests/test_allocation.py training/tests/test_source.py
test-training:
	@echo "Run optional CPU training and search regression tests with temporary data"
	@training/.venv/bin/python -m pytest -q training/tests players/search/tests
test-site:
	@echo "Validate the committed aggregate, type-check and build the Astro site"
	@python3 scripts/site_progress.py validate
	@npm --prefix site run build
lint-python:
	@echo "Check Python source with Ruff"
	@.venv/bin/ruff check engine/bindings/tests engine/tools players/llm players/search training/brass_rl training/scripts training/tests scripts site/scripts
lint: lint-python
	@echo "Check Rust formatting and Clippy warnings"
	@cargo fmt --all -- --check
	@cargo clippy --workspace --all-targets --locked -- -D warnings
check-public:
	@echo "Audit every tracked and candidate public path"
	@python3 scripts/check_public.py
play:
	@echo "Start the local play interface"
	@if test -x gui/play; then GUI_PORT="$(GUI_PORT)" exec gui/play; else echo "The optional play interface is available in the private workspace."; fi
site-dev:
	@echo "Refresh and serve the site on loopback port $(SITE_PORT)"
	@python3 scripts/site_dev.py --port "$(SITE_PORT)"
site-data:
	@echo "Refresh and validate the public research aggregates"
	@python3 scripts/site_progress.py data
publish-progress:
	@echo "Validate site data, review the diff; CONFIRM=1 permits a local commit"
	@python3 scripts/site_progress.py publish $(CONFIRM_FLAG)
release-model:
	@echo "Build a local portable weight archive and a model card"
	@python3 scripts/distributions.py model "$(MODEL)"
dataset-snapshot:
	@echo "Build a local game archive, datasheet and checksums"
	@python3 scripts/distributions.py dataset
release:
	@echo "Check the version, citation and changelog; CONFIRM=1 permits a local annotated tag"
	@python3 scripts/release.py "$(VERSION)" $(CONFIRM_FLAG)
sync-check:
	@echo "Confirm the private repositories and local data stay outside public Git"
	@python3 scripts/sync_check.py
.PHONY: test-oracle
test-oracle:
	@echo "Check the optional separately supplied oracle"
	@sh scripts/test_oracle.sh "$(or $(ORACLE),ts)"
