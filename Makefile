.PHONY: install test lint demo bench serve ui data cwru
install:
	pip install -e ".[dev]"
test:
	python -m pytest
lint:
	ruff check src tests examples benchmarks
demo:
	python examples/lifecycle_demo.py mlops_state
bench:
	python benchmarks/run_benchmark.py
serve:
	PREDMAINT_STATE=mlops_state uvicorn predmaint.serve:get_app --factory --port 8001
ui:
	mlflow ui --backend-store-uri sqlite:///mlops_state/mlflow.db
data:
	sh scripts/fetch_cwru_mirror.sh data/cwru
cwru:
	python examples/cwru_lifecycle.py data/cwru mlops_state_cwru
