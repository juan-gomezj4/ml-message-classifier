.PHONY: tests help install_env init_git pre-commit_update test check clean_env switch_main clean_branchs lint

####----Basic configurations----####

install_env: ## Install libs with UV and pre-commit
	@echo "🚀 Creating virtual environment using UV"
	uv sync --all-groups
	@echo "🚀 Installing pre-commit..."
	uv run pre-commit install

init_git: ## Initialize git repository
	@echo "🚀 Initializing local git repository..."
	git init -b main
	git add .
	git commit -m "🎉 Initial commit"

####----Tests----####
test: ## Run pytest with coverage
	@echo "🚀 Running tests"
	uv run pytest --cov

test_verbose: ## Run tests in verbose mode
	uv run pytest -v --no-header --cov

test_coverage: ## Generate XML coverage report
	uv run pytest --cov --cov-report xml:coverage.xml

####----Pre-commit----####
pre-commit_update: ## Update pre-commit hooks
	uv run pre-commit clean
	uv run pre-commit autoupdate

####----Clean----####
clean_env: ## Delete virtual environment
	@[ -d .venv ] && rm -rf .venv || echo ".venv directory does not exist"

####----Git----####
switch_main: ## Switch to main branch and pull
	git switch main
	git pull

clean_branchs: ## Delete local branches merged in origin
	git fetch -p
	for branch in $$(git for-each-ref --format '%(refname:short)' refs/heads/ | grep -v '^\*' | grep -v ' main$$'); do \
		if ! git show-ref --quiet refs/remotes/origin/$$branch; then \
			echo "Deleting local branch $$branch"; \
			git branch -D $$branch; \
		fi \
	done

####----Checks----####
check: ## Run pre-commit on all files
	uv run pre-commit run -a

lint: ## Run only Ruff
	uv run pre-commit run ruff

####----Project----####
help:
	@printf "%-30s %s\n" "Target" "Description"
	@printf "%-30s %s\n" "-----------------------" "----------------------------------------------------"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' Makefile | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
