
help:  ## Show help
	@grep -E '^[.a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

clean: ## Clean autogenerated files and egg-info
	rm -rf dist
	find . -type f -name "*.DS_Store" -ls -delete
	find . | grep -E "(__pycache__|\.pyc|\.pyo|\.egg-info)" | xargs rm -rf
	find . | grep -E ".pytest_cache" | xargs rm -rf
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf
	rm -f .coverage


clean-logs: ## Clean logs
	rm -rf logs/**
clean-output: ## Clean output
	rm -rf outputs
clean-lit-logs: ## Clean lit logs
	rm -rf lightning_logs

format: ## Run pre-commit hooks
	pre-commit run -a

sync-git: ## Merge changes from main branch to your current branch
	git pull
	git pull origin main

test: ## Run not slow tests
	pytest -k "not slow"

test-full: ## Run all tests
	pytest

train: ## Train the model
	python src/train.py

# clean cache
clean-cache:
	rm -rf __pycache__/

clean-slurm:
	rm -rf slurm-*
	
clean-all : clean clean-logs clean-output clean-lit-logs clean-cache ## Clean all

# rsync to remote server  
exportenv:
	export $(<.env grep -v "^#" | xargs)


REMOTE_SERVER=vsc10630@login.hpc.vub.be
REMOTE_PATH=/scratch/brussel/106/vsc10630/ssl_tuning_ddl

sync-to-cluster:
	echo $(REMOTE_SERVER)
	echo $(REMOTE_PATH)
	rsync -avz --exclude=.rsyncignore ./ $(REMOTE_SERVER):$(REMOTE_PATH)

fetch-logs: ## Fetch log files from the cluster
	mkdir -p local_logs
	rsync -avz $(REMOTE_SERVER):$(REMOTE_PATH)/logs/ ./local_logs/
