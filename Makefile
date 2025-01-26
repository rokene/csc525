# CONFIG ######################################################################

.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

PP=$(CURRENT_DIR)/portfolio-project
PP_APP=app.py

KNN_IRIS=$(CURRENT_DIR)/knn-iris
KNN_IRIS_APP=app.py


# PYTHON CONFIG ###############################################################

# ubuntu

# PYTHON_CONFIG=python3
# PYTHON_PIP_CONFIG=pip
# VNV_ACTIVATE=venv/bin/activate

# windows

PYTHON_CONFIG=py.exe
PYTHON_PIP_CONFIG=py.exe -m pip
VNV_ACTIVATE=venv/Scripts/activate

# TARGETS #####################################################################

.PHONY: help
help:
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*' $(MAKEFILE_LIST) | sort

.PHONY: pp-setup
pp-setup: ## setup dependencies and precursors for portfolio project
	@echo "pp: setting up portfolio project virtual env"
	@cd $(PP) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: pp
pp-draw: ## executes portfolio project Annotation Draw
	@echo "pp: starting portfolio project annotation drawing"
	@cd $(PP) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(PP)/$(PP_APP)
	@echo "pp: completed portfolio project annotation drawing"

.PHONY: knn-iris-setup
knn-iris-setup: ## setup dependencies and precursors for portfolio project
	@echo "pp: setting up portfolio project virtual env"
	@cd $(KNN_IRIS) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: knn-iris
knn-iris: ## executes portfolio project Annotation Draw
	@echo "pp: starting portfolio project annotation drawing"
	@cd $(KNN_IRIS) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(KNN_IRIS)/$(KNN_IRIS_APP)
	@echo "pp: completed portfolio project annotation drawing"

