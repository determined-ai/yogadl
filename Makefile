all: check test

check: black flake8 mypy

black:
	black --check yogadl
	black --check tests

flake8:
	flake8 yogadl
	flake8 tests

mypy:
	mypy yogadl
	mypy tests

fmt:
	black .

TEST_EXPR ?= ""

test-unit-gcp:
	pytest -v -k $(TEST_EXPR) tests/unit/gcp

test-unit-local:
	pytest -v -k $(TEST_EXPR) tests/unit/local

test-integration-gcp:
	pytest -v -k $(TEST_EXPR) tests/integration/gcp

test-integration-local:
	pytest -v -k $(TEST_EXPR) tests/integration/local

test-local: test-unit-local test-integration-local

test-gcp: test-unit-gcp test-integration-gcp

test: test-local

test-all: test-local test-gcp
