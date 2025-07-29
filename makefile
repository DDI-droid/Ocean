# Ocean Makefile (using uv and per-repo branches)

SHELL := /bin/bash
.ONESHELL:

# === Configuration ===
UV            := uv
REPO_BRANCHES := pufferlib:ocr_3.0
INSTALL_BRANCHES := pufferlib:ocr_3.0
GIT_BASE      := https://github.com/DDI-droid
ENV_NAME      := env
PYTH_VERSION  := 3.12

# === Colors ===
RESET  := \033[0m
RED    := \033[0;31m
GREEN  := \033[0;32m
YELLOW := \033[1;33m
BLUE   := \033[1;34m

# === Phony Targets ===
.PHONY: all env clone install-internal clean help setup clean-experiments clean-all

help:
	@printf '%b\n' ""
	@printf '%b\n' "$(BLUE)Usage: make <target>$(RESET)"
	@printf '%b\n' ""
	@printf '%b\n' "$(YELLOW)Targets:$(RESET)"
	@printf '  %b%-17s%b %s\n' "$(GREEN)" "env" "$(RESET)" "Create/update Python env and install deps"
	@printf '  %b%-17s%b %s\n' "$(GREEN)" "clone" "$(RESET)" "Wire up subtree remotes and fetch"
	@printf '  %b%-17s%b %s\n' "$(GREEN)" "install-internal" "$(RESET)" "Install internal repos editable via pip"
	@printf '  %b%-17s%b %s\n' "$(GREEN)" "setup" "$(RESET)" "Run env, clone, and install-internal"
	@printf '  %b%-17s%b %s\n' "$(RED)"   "clean" "$(RESET)" "Remove env and subtrees"
	@printf '%b\n' ""

env:
	@printf '%b\n' "$(BLUE)→ Creating/updating Python virtual environment with uv...$(RESET)"
	$(UV) venv $(ENV_NAME) --python $(PYTH_VERSION)	
	@printf '%b\n' "$(BLUE)→ Installing third-party dependencies...$(RESET)"
	. $(ENV_NAME)/bin/activate && $(UV) pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121
	. $(ENV_NAME)/bin/activate && $(UV) pip install -r requirements.txt
	@printf '%b\n' "$(GREEN)✓ Environment setup complete.$(RESET)"

clone:
	@printf '%b\n' "$(BLUE)→ Cloning sub-repositories...$(RESET)"
	@for rb in $(REPO_BRANCHES); do \
	  repo=$${rb%%:*}; branch=$${rb##*:}; \
	  printf '%b   • Cloning %s…%b\n' "$(YELLOW)" "$$repo" "$(RESET)"; \
	  git clone --recurse-submodules -b $$branch --single-branch $(GIT_BASE)/$$repo.git; \
	done
	@printf '%b\n' "$(GREEN)✓ Cloned sub-repositories.$(RESET)"

build:
	@printf '%b\n' "$(BLUE)→ Installing internal packages in editable mode...$(RESET)"
	@for rb in $(INSTALL_BRANCHES); do \
	  repo=$${rb%%:*}; \
	  printf '%b   • Installing %s…%b\n' "$(YELLOW)" "$$repo" "$(RESET)"; \
	  	. $(ENV_NAME)/bin/activate && cd $$repo && export TORCH_CUDA_ARCH_LIST="8.6" && $(UV) pip install --no-build-isolation -v -e .[train,atari]; \
	done
	@printf '%b\n' "$(GREEN)✓ Internal packages installed.$(RESET)"

setup: env clone build
	@printf '%b\n' "$(GREEN)✓ Ocean full setup complete!$(RESET)"

clean:
	@printf '%b\n' "$(RED)→ Cleaning up environment and subtrees...$(RESET)"
	rm -rf $(ENV_NAME)
	@for rb in $(REPO_BRANCHES); do \
	  repo=$${rb%%:*}; \
	  printf '%b   • Removing %s…%b\n' "$(YELLOW)" "$$repo" "$(RESET)"; \
	  rm -rf $$repo; \
	done
	rm -rf resources
	@printf '%b\n' "$(GREEN)✓ Cleaned.$(RESET)"

clean-experiments:
	@printf '%b\n' "$(RED)→ Cleaning up experiments...$(RESET)"
	rm -rf experiments
	rm -rf .neptune
	@printf '%b\n' "$(GREEN)✓ Experiments cleaned.$(RESET)"

clean-all: clean clean-experiments
	@printf '%b\n' "$(GREEN)✓ All cleaned.$(RESET)"