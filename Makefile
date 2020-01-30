all: check

check: black flake8 mypy

black:
	black --check yogadl

flake8:
	flake8 yogadl

mypy:
	mypy yogadl

fmt:
	$ black .

TEST_EXPR ?= ""

test:
	pytest -v -k $(TEST_EXPR) tests/
