.PHONY: setup test lint typecheck format run-eval clean
setup:
	pip install -e '.[dev]'
test:
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=src pytest -q
lint:
	ruff check src tests
typecheck:
	mypy src
format:
	ruff format src tests
run-eval:
	pref-lab evaluate --config configs/local.yaml
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache outputs
