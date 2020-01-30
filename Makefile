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

test:
	pytest -v -k $(TEST_EXPR) tests/
