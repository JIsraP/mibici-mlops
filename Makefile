#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = mibici_mlops
PYTHON_VERSION = 3.11
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
	$(PYTHON_INTERPRETER) -m mibici_mlops.dataset download

## Limpiar viajes y registrar v0 y v1 en MLflow (variantes estricta y laxa)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m mibici_mlops.dataset main --variant estricta
	$(PYTHON_INTERPRETER) -m mibici_mlops.dataset main --variant laxa

## Entrada del modelo desde la version oficial con banderas (registra v1_modelo_oficial)
.PHONY: oficial
oficial:
	$(PYTHON_INTERPRETER) -m mibici_mlops.model_input

## Construir demanda y variables, registrar v2 y v3
.PHONY: features
features:
	$(PYTHON_INTERPRETER) -m mibici_mlops.features --variant oficial
	$(PYTHON_INTERPRETER) -m mibici_mlops.features --variant estricta
	$(PYTHON_INTERPRETER) -m mibici_mlops.features --variant laxa

## Validar la tabla de demanda contra los viajes
.PHONY: validate
validate:
	$(PYTHON_INTERPRETER) -m mibici_mlops.validation --variant oficial

## Lineas base contra la version oficial (experimento de la Fase 2)
.PHONY: baseline
baseline:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.baselines

## Comparar lineas base entre variantes de limpieza (experimento de la Fase 1)
.PHONY: baseline-variants
baseline-variants:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.baselines --variant estricta --experiment MiBici_Fase1_Experimentos_Datos
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.baselines --variant laxa --experiment MiBici_Fase1_Experimentos_Datos

## Ajustar y entrenar los modelos de demanda de la Fase 2
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.train

## Registrar en MLflow el mejor modelo que cumple los criterios de la Fase 2
.PHONY: register
register:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.train --register-only

## Tabla y figuras comparativas de todas las corridas del experimento de modelos
.PHONY: compare
compare:
	$(PYTHON_INTERPRETER) -m mibici_mlops.modeling.compare

## Figuras del EDA y EDA de viajes marcados
.PHONY: plots
plots:
	$(PYTHON_INTERPRETER) -m mibici_mlops.plots --variant oficial
	$(PYTHON_INTERPRETER) -m mibici_mlops.flagged_trips

## Tuberia completa (la version oficial la genera el data engineer en data/external/version_oficial)
.PHONY: pipeline
pipeline: download data oficial features validate baseline plots


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
