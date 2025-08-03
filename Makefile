# Makefile for ML Pipeline Project

.PHONY: help install test test-unit test-integration test-fast test-slow test-cov clean lint format check-deps

# Default target
help:
	@echo "Available commands:"
	@echo "  install          Install project dependencies"
	@echo "  test             Run all tests"
	@echo "  test-unit        Run only unit tests"
	@echo "  test-integration Run only integration tests"
	@echo "  test-fast        Run fast tests (excludes slow tests)"
	@echo "  test-slow        Run only slow tests"
	@echo "  test-cov         Run tests with coverage report"
	@echo "  test-parallel    Run tests in parallel"
	@echo "  lint             Run code linting"
	@echo "  format           Format code with black"
	@echo "  check-deps       Check for security vulnerabilities in dependencies"
	@echo "  clean            Clean up generated files"

# Installation
install:
	pip install --upgrade pip
	pip install -r requirements.txt

install-dev:
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install black flake8 mypy bandit safety

# Testing
test:
	pytest

test-unit:
	pytest -m "unit"

test-integration:
	pytest -m "integration"

test-fast:
	pytest -m "not slow"

test-slow:
	pytest -m "slow"

test-cov:
	pytest --cov=src --cov-report=html --cov-report=term

test-parallel:
	pytest -n auto

test-specific:
	@echo "Usage: make test-specific TEST=tests/test_specific.py::TestClass::test_method"
	pytest $(TEST) -v

# Code Quality
lint:
	flake8 src/ tests/ main.py --max-line-length=120 --ignore=E203,W503
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/ main.py --line-length=120

format-check:
	black src/ tests/ main.py --line-length=120 --check

# Security
check-deps:
	safety check
	bandit -r src/ -f json

# Cleanup
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf .pytest_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	rm -rf coverage.xml
	rm -rf tests/data/
	rm -rf tests/logs/
	rm -rf tests/models/

# Pipeline specific commands
test-db:
	pytest tests/test_database.py -v

test-preprocessing:
	pytest tests/test_preprocessing.py -v

test-training:
	pytest tests/test_training.py -v

test-evaluation:
	pytest tests/test_evaluation.py -v

test-utils:
	pytest tests/test_utils.py -v

test-dataset:
	pytest tests/test_dataset.py -v

test-e2e:
	pytest tests/test_integration.py -v

# Coverage by module
cov-db:
	pytest tests/test_database.py --cov=src.db --cov-report=term

cov-preprocessing:
	pytest tests/test_preprocessing.py --cov=src.preprocessing --cov-report=term

cov-training:
	pytest tests/test_training.py --cov=src.training --cov-report=term

cov-evaluation:
	pytest tests/test_evaluation.py --cov=src.training.evaluation --cov-report=term

# Development helpers
watch-tests:
	@echo "Watching for changes and running tests..."
	@echo "Install 'pytest-watch' with: pip install pytest-watch"
	ptw

install-pre-commit:
	pip install pre-commit
	pre-commit install

# Documentation
docs:
	@echo "Generating documentation..."
	@echo "TODO: Add documentation generation command"

# Docker support (if needed)
docker-test:
	docker run --rm -v $(PWD):/app -w /app python:3.9 make install test

# Performance testing
test-performance:
	pytest tests/test_integration.py::TestPipelineRobustness::test_pipeline_memory_efficiency -v

# Generate test data
generate-test-data:
	python -c "from tests.conftest import sample_dataframe; import pandas as pd; df = sample_dataframe(); df.to_parquet('tests/sample_data.parquet')"

# Quick smoke test
smoke-test:
	pytest tests/test_utils.py::TestIOUtils::test_load_config_success -v