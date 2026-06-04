# Put it first so that "make" without argument is like "make help".
ROOTDIR = $(CURDIR)
version_type := local
EXTRA_ARGS =
PIP_ARGS =
BUILD_ARGS =
# Personal overrides (e.g. PIP, RUN). See .env.example.
-include .env
# Package installer. Default: pip. Override in .env: PIP=uv pip --python python3 --system
PIP ?= pip
# Command runner. Default: empty (run directly).
RUN ?=
COMMIT_UNIXTIME := $(shell git log -n 1 --pretty='format:%ct')
COMMIT_DATETIME := $(shell date -d @${COMMIT_UNIXTIME} +'%Y%m%d%H%M%S')
COMMIT_ID := $(shell git rev-parse --short HEAD)

ifeq (${version_type}, beta)
	VERSION_POSTFIX := b${COMMIT_DATETIME}
	DEPLOY_ENV = dev
else ifeq (${version_type}, dev)
	VERSION_POSTFIX := .dev${COMMIT_DATETIME}
	DEPLOY_ENV = dev
else ifeq (${version_type}, rc)
	VERSION_POSTFIX := .rc${COMMIT_DATETIME}
	DEPLOY_ENV = test
else ifeq (${version_type}, local)
	VERSION_POSTFIX := +local${COMMIT_DATETIME}.${COMMIT_ID}
else ifeq (${version_type}, release)
	VERSION_POSTFIX :=
	DEPLOY_ENV = prod
else
	ERR_MSG = version_type must be one of beta, dev, rc, local, release
endif

version:
	@if [ -n "${ERR_MSG}" ]; then echo ${ERR_MSG}; exit 1; fi
	@echo commit_time: ${COMMIT_DATETIME}
	@echo commit_id: ${COMMIT_ID}
	@echo version_postfix: ${VERSION_POSTFIX}
	@echo version_type: ${version_type}
	@echo ${VERSION_POSTFIX} > VERSION_POSTFIX

install: version
	$(PIP) install .${EXTRA_ARGS} ${BUILD_ARGS} ${PIP_ARGS}

install-editable: version
	$(PIP) install --config-settings editable_mode=compat -e .${EXTRA_ARGS} ${BUILD_ARGS} ${PIP_ARGS}

dev-env:
	@$(PIP) install -r scm/requirements.txt ${PIP_ARGS}
	@$(PIP) install "lerobot>=0.4.0" --no-deps ${PIP_ARGS}
	@$(RUN) pre-commit install

auto-format:
	$(RUN) python3 scm/lint/check_lint.py --auto_format

check-lint:
	@$(RUN) python3 scm/lint/check_lint.py
	@$(RUN) pre-commit run check-merge-conflict
	@$(RUN) pre-commit run check-license-header --all-files

dist-build: version
	@mkdir -p build/dist
	@python3 -m pip wheel . --wheel-dir=build/dist --no-deps

doc:
	make -C docs html

# Debug a small AutoAPI docs subset without running the full docs build.
# Example: make doc-debug-api API_TARGETS="robo_orchard_lab/version.py"
doc-debug-api:
	make -C docs debug-api API_TARGETS="$(API_TARGETS)"

# Debug one or more tutorial gallery scripts without running the full docs
# build. Example:
# make doc-debug-tutorial TUTORIAL_TARGETS="tutorials/model_zoo_tutorial/nonb-02_inference_api.py"
doc-debug-tutorial:
	make -C docs debug-tutorial TUTORIAL_TARGETS="$(TUTORIAL_TARGETS)"

# Debug overview docs without AutoAPI or tutorial generation.
doc-debug-overview:
	make -C docs debug-overview

doc-clean:
	make -C docs clean

test:
	make -C tests RUN="$(RUN)"

test_ut:
	make -C tests test_ut RUN="$(RUN)"

test_it:
	make -C tests test_it RUN="$(RUN)"

show-args:
	@echo "PIP_ARGS: $(PIP_ARGS)"
	@echo "BUILD_ARGS: $(BUILD_ARGS)"
	@echo "EXTRA_ARGS: $(EXTRA_ARGS)"
