all: check

check: black flake8 mypy

black:
	black --check yogadl

flake8:
	flake8 yogadl

mypy:
	mypy yogadl
