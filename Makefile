# Bundeskanzler KI Makefile
# Vereinfacht Entwicklungsworkflows

# Python executable in venv
PYTHON = /home/tobber/bkki_venv/bin/python
PIP = /home/tobber/bkki_venv/bin/pip

.PHONY: help install test test-real coverage clean lint format

# Default target
help:
	@echo "Bundeskanzler KI Development Commands:"
	@echo ""
	@echo "  install     - Install all dependencies"
	@echo "  test        - Run all tests"
	@echo "  test-working - Run verified working tests (recommended)"
	@echo "  test-clean   - Run clean tests with real dependencies" 
	@echo "  test-memory  - Test hierarchical memory system"
	@echo "  demo-memory  - Run memory system demonstration"
	@echo "  coverage    - Run tests with coverage report"
	@echo "  clean       - Clean temporary files"
	@echo "  lint        - Run code linting"
	@echo "  format      - Format code"
	@echo "  freeze      - Update requirements.txt"

# Installation
install:
	$(PIP) install -r requirements-clean.txt

install-dev:
	$(PIP) install -r requirements-clean.txt
	$(PIP) install black flake8 mypy

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test-real:
	$(PYTHON) -m pytest tests/test_validation_real.py -v

test-clean:
	$(PYTHON) -m pytest tests/test_validation_clean.py -v

test-memory:
	$(PYTHON) -m pytest tests/test_hierarchical_memory.py -v

demo-memory:
	$(PYTHON) demo_memory.py

test-working:
	$(PYTHON) -m pytest tests/test_transformer_model.py tests/test_transfer_learning.py -v

# Erweiterte Test-Targets
test-unit:
	$(PYTHON) -m pytest tests/ -m "unit" -v

test-integration:
	$(PYTHON) -m pytest tests/ -m "integration" -v

test-api:
	$(PYTHON) -m pytest tests/ -m "api" -v

test-memory:
	$(PYTHON) -m pytest tests/ -m "memory" -v

test-security:
	$(PYTHON) -m pytest tests/ -m "security" -v

test-performance:
	$(PYTHON) -m pytest tests/ -m "performance" -v --benchmark-only

test-slow:
	$(PYTHON) -m pytest tests/ -m "slow" -v

test-parallel:
	$(PYTHON) -m pytest tests/ -n auto -v

test-docker:
	docker-compose --profile gpu --profile with-db --profile with-redis --profile testing run --rm test-runner

coverage:
	$(PYTHON) -m pytest --cov=. --cov-report=html --cov-report=term --cov-report=xml
	@echo "Coverage report: htmlcov/index.html"

coverage-report:
	$(PYTHON) -m pytest --cov=. --cov-report=html
	open htmlcov/index.html

# Load Testing
load-test:
	locust -f tests/load_tests.py --host=http://localhost:8000

# Security Testing
security-test:
	$(PYTHON) -m pytest tests/ -m "security" -v
	bandit -r . -x tests/

# Performance Benchmarking
benchmark:
	$(PYTHON) -m pytest tests/ -m "performance" --benchmark-only --benchmark-save=benchmarks

benchmark-compare:
	$(PYTHON) -m pytest tests/ -m "performance" --benchmark-only --benchmark-compare

# Code quality
lint:
	flake8 . --exclude=lib,bin,include,share
	mypy . --ignore-missing-imports

format:
	black . --exclude="/(lib|bin|include|share)/"

# Maintenance
clean:
	rm -rf __pycache__ .pytest_cache htmlcov .coverage
	find . -name "*.pyc" -delete
	find . -name "*.pyo" -delete

freeze:
	$(PIP) freeze > requirements.txt

# Development server (for future API)
# serve:
# 	uvicorn bundeskanzler_ki:app --reload --host 0.0.0.0 --port 8000

# Docker (for future deployment)
# docker-build:
# 	docker build -t bundeskanzler-ki .

# docker-run:
# 	docker run -p 8000:8000 bundeskanzler-ki