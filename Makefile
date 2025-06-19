# This is a modified version of the original PyLLMD makefile

TOOLKIT_PROFILE ?= yul201
NOW := $(shell date +%s)
TOOLKIT_ACCOUNT_ID ?= $(shell eai account get --profile $(TOOLKIT_PROFILE) --no-header --field fullName)
USER := $(shell eai user get --profile ${TOOLKIT_PROFILE} --field name)

# Depending on the profile, the home data name will be different.
ifeq ($(TOOLKIT_PROFILE), yul101)
    USER := $(shell eai account get --field name --no-header)
    HOME_DATA_NAME := ${TOOLKIT_ACCOUNT_ID}.home
    ACCOUNT := llmd
    ORG := snow
    RESULTS_MOUNT_PATH := /mnt/$(ACCOUNT)/results
    DATA_MOUNT_PATH := /mnt/$(ACCOUNT)/data
    BASE_MODELS_MOUNT_PATH := /mnt/$(ACCOUNT)/base_models
    IMAGE := registry.toolkit-sp.yul201.service-now.com/snow.research.tapes/interactive-toolkit:throughput
    IMAGE_REVISION ?= latest
else ifeq ($(TOOLKIT_PROFILE), yul201)
    USER := $(shell eai user get --profile ${TOOLKIT_PROFILE} --field name)
    HOME_DATA_NAME := snow.home.${USER}
    ACCOUNT := tapes
    ORG := snow.research
    RESULTS_MOUNT_PATH := /mnt/llmd/results
    DATA_MOUNT_PATH := /mnt/llmd/data
    BASE_MODELS_MOUNT_PATH := /mnt/llmd/base_models
    IMAGE := registry.toolkit-sp.yul201.service-now.com/snow.research.tapes/interactive_toolkit
    IMAGE_REVISION ?= apiche #TODO:
else
   $(error Unsupported TOOLKIT_PROFILE: $(TOOLKIT_PROFILE))
endif

JOB_ACCOUNT ?= $(ORG).$(ACCOUNT) # snow_infiniband.pyllmd
BID ?= 1


TRANSFORMERS_CACHE_DATA := $(ORG).$(ACCOUNT).transformers_cache
DATA_OBJ := $(ORG).$(ACCOUNT).data
RESULTS_OBJ := $(ORG).$(ACCOUNT).results
BASE_MODELS_OBJ := $(ORG).$(ACCOUNT).base_models

# Text Generation Inference v2.0.4
TGI_IMAGE ?= ghcr.io/huggingface/text-generation-inference:sha-f426a33

BIGCODE_JOB_NAME ?= $(ORG)_$(USER)_$(ACCOUNT)_bigcode_job
BIGCODE_OLD_JOB_NAME := $(ORG)_$(USER)_$(ACCOUNT)_bigcode_old_job_${NOW}
BIGCODE_MODEL_ID := bigcode/starcoderbase

LLAMA_JOB_NAME ?= $(ORG)_$(USER)_$(ACCOUNT)_llama_65b_job
LLAMA_OLD_JOB_NAME := $(ORG)_$(USER)_$(ACCOUNT)_llama_65b_old_job_${NOW}
LLAMA_RM_JOB_NAME ?= $(ORG)_$(USER)_$(ACCOUNT)_llama_rm
LLAMA_RM_OLD_JOB_NAME := $(ORG)_$(USER)_$(ACCOUNT)_llama_rm_old_job_${NOW}
FALCON_RM_JOB_NAME ?= $(ORG)_$(USER)_$(ACCOUNT)_falcon_rm
FALCON_RM_OLD_JOB_NAME := $(ORG)_$(USER)_$(ACCOUNT)_falcon_rm_old_job_${NOW}
# LabelStudio refuses to work if the URL contains underscores
LABEL_STUDIO_JOB_NAME ?= labelstudio
LABEL_STUDIO_OLD_JOB_NAME ?= labelstudio_old_job_${NOW}
LABEL_STUDIO_IMAGE := heartexlabs/label-studio:latest
# Dialogue authoring tool
DIALOGUE_AUTHOR_JOB_NAME ?= dialogue_author
DATA_ASSIST_JOB_NAME ?= data_assist

CPU ?= 6
CPU_MEM ?= 128
GPU ?= 1
GPU_MEM ?= 80

# Accelerate settings
NPROC ?= 1
FP ?= bf16

RAND_ID := $(shell python -c 'import random; print(random.randint(0, int(1e9)))')
DATE_ID := $(shell date +%Y_%m_%d__%H_%M_%S)
JOB_NAME ?= $(USER)$(RAND_ID)
OLD_JOB_NAME := $(JOB_NAME)_old_${NOW}

TUNING_RESULTS_DIR ?= /home/toolkit/tuning_results
TUNING_DATA_DIR ?=  /home/toolkit/tuning_data
TUNING_DATA_CONFIG :=
ifeq ($(TUNING_DATA_CONFIG),)
	_DATA_PARAMS := finetune/data=simple_tuning finetune.data.data_parts_train.0.path=$(TUNING_DATA_DIR)/train finetune.data.data_parts_valid.0.path=$(TUNING_DATA_DIR)/dev
else
	_DATA_PARAMS := finetune/data=$(TUNING_DATA_CONFIG)
endif

TEXT_GEN_INFERENCE_ENV ?= text-gen-inference
# Another option "Tesla T4". You will need `GPU=4` to use that. 
GPU_TYPE ?= "A100"
MODEL_ID ?= bigcode/santacoder
MAX_RUN_TIME ?= 7200

# a function to create a git snapshot based on the active commit
# usage: $(call create_snapshot)
define create_snapshot
	@( [ -d $(_WORKDIR) ] && \
	cd $(_WORKDIR) && \
	[ `git rev-parse HEAD` = $(REVISION) ] && \
	[ -z "`git status -s`" ] && \
	echo "using existing snapshot" ) \
	|| \
	( echo "creating a new snapshot ..." && \
	rm -rf $(_WORKDIR) && \
	git clone --recurse-submodules git@github.com:ServiceNow/PipelineRL-SWE.git $(_WORKDIR) && \
	cd $(_WORKDIR) && \
	git checkout $(REVISION) && \
	git submodule update && \
	echo "snapshot created successfully")
endef

# a function to rename a job
# usage: $(call rename_job,account,old_name,new_name)
define rename_job
	@id=`eai job ls --account $(JOB_ACCOUNT) -N $(2) -n1 --field id --state all`;\
	[ -z $${id} ] || eai job set $${id} --name $(3)
endef

# a function to kill a job
# usage: $(call kill_job,account,name)
define kill_job
	@id=`eai job ls --account $(1) -N $(2) -n1 --field id --state alive`;\
	[ -z $${id} ] || (\
		eai job kill `eai job ls --account $(1) -N $(2) -n1 --field id`;\
		while [ `eai job ls --account $(1) -N $(2) -n1 --field state` != "CANCELLED" ];\
		do\
			echo "Waiting for your old job to cancel...";\
			sleep 5; \
		done;\
	)
endef

# a function to wait for a job to be RUNNING
# usage: $(call wait_for_job_running,account,name)
define wait_for_job_running
	@id=`eai job ls --account $(1) -N $(2) -n1 --field id --state all`;\
	[ -z $${id} ] || (\
		while [ `eai job ls --account $(1) -N $(2) -n1 --field state` != "RUNNING" ];\
		do\
			echo "Waiting for your job to start...";\
			sleep 5; \
		done;\
		echo "Job running as $${id}";\
	)
endef


# a function to get information about a job
# usage: $(call get_job_info,account,name)
define get_job_info
	@id=`eai job ls --account $(JOB_ACCOUNT) -N $(2) -n1 --field id`;\
	[ -z $${id} ] && echo "No job found" || (\
		eai job info $${id};\
	)
endef

# a function to get the logs of a job
# usage: $(call get_job_logs,account,name)
define get_job_logs
	@id=`eai job ls --account $(JOB_ACCOUNT) -N $(2) -n1 --field id`;\
	[ -z $${id} ] && echo "No job found" || (\
		eai job logs -f $${id};\
	)
endef

REVISION ?= $(shell git rev-parse HEAD)
SNAPSHOT ?= 1
ifeq ($(SNAPSHOT),1)
	_WORKDIR := /home/toolkit/snapshots2/$(REVISION)
else
	_WORKDIR := $(PWD)
endif

LOCAL ?= 0  # Use LOCAL=1 for local execution of the latest snapshot
DRY_RUN ?= 0  # Use DRY_RUN=1 to print the commands instead of executing them
# define DRY_RUN_PREFIX
ifeq ($(DRY_RUN), 1)
	_DRY_RUN_PREFIX := @echo '
	_DRY_RUN_SUFFIX := '
else
	_DRY_RUN_PREFIX :=
	_DRY_RUN_SUFFIX :=
endif
_RED=\033[0;31m
_NO_COLOR=\033[0m

# `make job` will use the Conda executable that CONDA_EXE points to.
# If you have a Conda environment activated,
# `make job` will use this environment for the launched job.
ENV ?= $(CONDA_DEFAULT_ENV)
# By default we launch the job in a Conda environment
CONDA ?= 1
# Optionally we launch the job using Huggingface Accelerate
ACCELERATE ?= 0
# Optionally we launch the job using Deepspeed (integrated in Huggingface Accelerate)
DEEPSPEED ?= 0
# To avoid users relying on a local configuration generated by `accelerate config`,
# we provide a default accelerate configuration file,
# which internally calls a deepspeed configuration file
ACCELERATE_CFG ?= conf/deepspeed/accelerate_base.yaml
ACCELERATE_LOCAL_CFG ?= conf/deepspeed/accelerate_local.yaml
DEEPSPEED_CFG ?= conf/deepspeed/deepspeed_stage3_bf16.json

_CONDA_PREFIX := $(CONDA_EXE) run -n $(ENV) --no-capture-output
ifeq ($(NPROC), 1)
	_ACCELERATE_PREFIX := accelerate launch --mixed_precision=$(FP) --config_file $(ACCELERATE_LOCAL_CFG)
else
	_ACCELERATE_PREFIX := accelerate launch --multi_gpu --mixed_precision=$(FP) --num_processes $(NPROC) --config_file $(ACCELERATE_CFG)
endif
_DEEPSPEED_PREFIX := accelerate launch --use_deepspeed --mixed_precision=$(FP) --num_processes $(NPROC) --config_file $(ACCELERATE_CFG) --deepspeed_config_file $(DEEPSPEED_CFG)

BASH_COMMAND := bash -c "$(COMMAND)"
_COMMAND = $(BASH_COMMAND)
_CONDA_COMMAND := $(_CONDA_PREFIX) bash -c "$(COMMAND)"
_ACCELERATE_COMMAND := $(_CONDA_PREFIX) $(_ACCELERATE_PREFIX) $(COMMAND)
_DEEPSPEED_COMMAND := $(_CONDA_PREFIX) $(_DEEPSPEED_PREFIX) $(COMMAND)

_DIALOGUE_AUTHOR_COMMAND := $(_CONDA_PREFIX) python -m scripts.dialogue_author_webapp
# requires having a virtualenv named sqlite-web with pip install sqlite-web
_DIALOGUE_AUTHOR_SQLITE_COMMAND := $(CONDA_EXE) run -n sqlite-web --no-capture-output sqlite_web -H 0.0.0.0 -p 8080 -x /mnt/llmd/data/dialogue_author/telemetry/dat_telemetry.db 

DATA_ASSIST_DB_PATH ?= "/mnt/llmd/data/dialogue_author/fsdb"
_DATA_ASSIST_COMMAND := $(_CONDA_PREFIX) python -m uvicorn --host 0.0.0.0 --port 8080 --workers 4 scripts.dialogue_author_webapp.failure_explorer:app

_CPU := $(CPU)
_CPU_MEM := $(CPU_MEM)
_GPU := $(GPU)

ifeq ($(CONDA), 1)
	_COMMAND := $(_CONDA_COMMAND)
# in this project we launch multi-gpu jobs using conda
	_CPU := $$(($(NPROC) * $(CPU)))
	_CPU_MEM := $$(($(NPROC) * $(CPU_MEM)))
	_GPU := $$(($(NPROC) * $(GPU)))
endif
ifeq ($(ACCELERATE), 1)
	_COMMAND := $(_ACCELERATE_COMMAND)
	_CPU := $$(($(NPROC) * $(CPU)))
	_CPU_MEM := $$(($(NPROC) * $(CPU_MEM)))
	_GPU := $$(($(NPROC) * $(GPU)))
endif
ifeq ($(DEEPSPEED), 1)
	_COMMAND := $(_DEEPSPEED_COMMAND)
	_CPU := $$(($(NPROC) * $(CPU)))
	_CPU_MEM := $$(($(NPROC) * $(CPU_MEM)))
	_GPU := $$(($(NPROC) * $(GPU)))
endif

_PYTHONPATH := $(_WORKDIR)

.PHONY: job
job:
ifndef COMMAND
	@echo "Must specify the command to run"
	exit 1
endif
ifeq ($(DEEPSPEED), 1)
ifeq ($(ACCELERATE), 1)
	printf "${_RED} ERROR: ACCELERATE=1 incompatible with DEEPSPEED=1! ${_NO_COLOR}\n"
	exit 1
endif
endif
ifeq ($(SNAPSHOT), 1)
	$(call create_snapshot)
else
ifneq ($(LOCAL),1)
	printf "${_RED} WARNING: strongly consider SNAPSHOT=1 for launching remote jobs! ${_NO_COLOR}\n"
endif
endif
ifeq ($(LOCAL), 1)
	$(_DRY_RUN_PREFIX) cd $(_WORKDIR) && \
	PYTHONPATH=${_PYTHONPATH} $(_COMMAND) $(_DRY_RUN_SUFFIX)
else
	$(call rename_job,$(ACCOUNT),$(JOB_NAME),$(OLD_JOB_NAME))
	$(_DRY_RUN_PREFIX) eai job submit \
		--name $(JOB_NAME) \
		--bid $(BID) \
		--account $(JOB_ACCOUNT) \
		--env HOME=/home/toolkit \
		--env PYTHONPATH=${_PYTHONPATH} \
		--env HF_DATASETS_CACHE=/transformers_cache \
		--env HF_HOME=/transformers_cache \
		--workdir $(_WORKDIR) \
		--image $(IMAGE):$(IMAGE_REVISION) \
		--data $(HOME_DATA_NAME):/home/toolkit \
		--data $(TRANSFORMERS_CACHE_DATA):/transformers_cache \
		--data $(DATA_OBJ):$(DATA_MOUNT_PATH) \
		--data $(RESULTS_OBJ):$(RESULTS_MOUNT_PATH) \
		--data $(BASE_MODELS_OBJ):$(BASE_MODELS_MOUNT_PATH) \
		--cpu $(_CPU) \
		--mem $(_CPU_MEM) \
		--gpu $(_GPU) \
		--gpu-mem $(GPU_MEM) \
		--restartable \
		-- $(_COMMAND) $(_DRY_RUN_SUFFIX)
endif

# This target use multinode.yaml to specify
# - a different base image, currently FastLLM-one
# - use all resources per-node
# - internal DNS settings
# It also differs from the "job:" target in that it runs your raw $(COMMAND)
.PHONY: multi-replica-job
multi-replica-job:
ifndef COMMAND
	@echo "Must specify the command to run"
	exit 1
endif
ifeq ($(SNAPSHOT), 1)
	$(call create_snapshot)
endif
	$(call rename_job,$(ACCOUNT),$(JOB_NAME),$(OLD_JOB_NAME))
	$(_DRY_RUN_PREFIX) eai job submit --replicas $(REPLICAS) \
		--name $(JOB_NAME) \
		--file multinode_rl.yaml \
		--bid $(BID) \
		--account $(JOB_ACCOUNT) \
		--env HOME=/home/toolkit \
		--env PATH=/home/toolkit/.conda/envs/$(ENV)/bin:/opt/conda/condabin:/home/toolkit/.local/bin:/home/toolkit/bin:/opt/conda/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
		--env PYTHONPATH=${_PYTHONPATH} \
		--env HF_DATASETS_CACHE=/transformers_cache \
		--env HF_HOME=/transformers_cache \
		--workdir $(_WORKDIR) \
		--data $(HOME_DATA_NAME):/home/toolkit \
		--data $(TRANSFORMERS_CACHE_DATA):/transformers_cache \
		--data $(DATA_OBJ):$(DATA_MOUNT_PATH) \
		--data $(RESULTS_OBJ):$(RESULTS_MOUNT_PATH) \
		--data $(BASE_MODELS_OBJ):$(BASE_MODELS_MOUNT_PATH) \
		-- $(BASH_COMMAND) $(_DRY_RUN_SUFFIX)

SETUP_ENV ?= prl

setup: 
	@echo "Setting up the environment"
	conda create -n $(SETUP_ENV) -y python=3.11
	conda run --no-capture-output -n $(SETUP_ENV) pip install TapeAgents[finetune]
	conda run --no-capture-output -n $(SETUP_ENV) pip install -e .