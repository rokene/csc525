# CONFIG ######################################################################

.DEFAULT_GOAL := help

CURRENT_DIR := $(CURDIR)

PP=$(CURRENT_DIR)/portfolio-project
PP_APP=app.py

KNN_IRIS=$(CURRENT_DIR)/knn-iris
KNN_IRIS_APP=app.py

LINEAR_REGRESSION_HOUSING=$(CURRENT_DIR)/linear-regression
LINEAR_REGRESSION_HOUSING_APP=app.py

TF_CNN_HR_DIGITS=$(CURRENT_DIR)/tensorflow-hr-digits
TF_CNN_HR_DIGITS_JUPYTER=app.ipynb

TEXT_AUG=$(CURRENT_DIR)/text-data-augment
TEXT_AUG_APP=$(TEXT_AUG)/app.py


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

.PHONY: linear-regression-housing-setup
linear-regression-housing-setup: ## setup dependencies and precursors for linear regression housing
	@echo "pp: setting up portfolio project virtual env"
	@cd $(LINEAR_REGRESSION_HOUSING) && $(PYTHON_CONFIG) -m venv venv && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_PIP_CONFIG) install --upgrade pip && \
		$(PYTHON_PIP_CONFIG) install -r requirements.txt

.PHONY: linear-regression-housing
linear-regression-housing: ## executes linear regression housing
	@echo "pp: starting portfolio project annotation drawing"
	@cd $(LINEAR_REGRESSION_HOUSING) && \
		. $(VNV_ACTIVATE) && \
		$(PYTHON_CONFIG) $(LINEAR_REGRESSION_HOUSING)/$(LINEAR_REGRESSION_HOUSING_APP)
	@echo "pp: completed portfolio project annotation drawing"

.PHONY: tensorflow-cnn-handwritten-digits-setup
tensorflow-cnn-handwritten-digits-setup: ## sets up the tensflow v2 cnn for handwritten digits
	@echo "pp: setting up tensorflow v2 handwritten cnn project"
	@cd $(TF_CNN_HR_DIGITS) && \
		$(PYTHON_CONFIG) -m pipenv install

.PHONY: tensorflow-cnn-handwritten-digits
tensorflow-cnn-handwritten-digits: ## executes tensorflow v2 cnn for handwritten digits
	@echo "pp: starting tensflow v2 cnn for handwritten digits"
	@cd $(TF_CNN_HR_DIGITS) && \
		$(PYTHON_CONFIG) -m pipenv run jupyter ./$(TF_CNN_HR_DIGITS_JUPYTER)
	@echo "pp: completed tensflow v2 cnn for handwritten digits"

.PHONY: text-augmentation-setup
text-augmentation-setup: ## sets up text data augmentation project
	@echo "pp: setting uptext data augmentation project"
	@cd $(TEXT_AUG) && \
		$(PYTHON_CONFIG) -m pipenv install
	@$(PYTHON_CONFIG) -m pipenv run 

.PHONY: text-augmentation
text-augmentation: ## executes text data augmentation project
	@echo "pp: starting text data augmentation project"
	@cd $(TEXT_AUG) && \
		$(PYTHON_CONFIG) -m pipenv run $(TEXT_AUG_APP)
	@echo "pp: completed text data augmentation project"
