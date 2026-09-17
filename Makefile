.PHONY: test validate lint

test:
	python -m unittest discover -s tests -v
	python -m compileall -q src plugins/dflash_vllm_patch

validate:
	python -m dflash_bench validate configs/*.toml

lint:
	ruff check src tests scripts plugins/dflash_vllm_patch
