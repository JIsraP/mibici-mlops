#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mibici-mlops
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Descargar los CSV de 2025 y el catalogo de estaciones
.PHONY: download
download:
	$(PYTHON_INTERPRETER) -m mibici.dataset download

## Limpiar viajes y registrar v0 y v1 en MLflow (variantes estricta y laxa)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici.dataset main --variant estricta
	$(PYTHON_INTERPRETER) -m mibici.dataset main --variant laxa

## Construir demanda y variables, registrar v2 y v3
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici.features --variant estricta
	$(PYTHON_INTERPRETER) -m mibici.features --variant laxa

## Comparar lineas base entre variantes de datos
.PHONY: baseline
baseline:
	$(PYTHON_INTERPRETER) -m mibici.modeling.train --variant estricta
	$(PYTHON_INTERPRETER) -m mibici.modeling.train --variant laxa

## Figuras del EDA
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) -m mibici.plots --variant estricta

## Tuberia completa
.PHONY: pipeline
pipeline: download data features baseline plots


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
