# uv python install 3.14t

environment:
	uv python install 3.14t
	uv venv --python 3.14t
	# source .venv/bin/activate
	# uv pip install -r requirements.txt
.PHONY: environment

run:
	PYTHON_GIL=0 uv run --python 3.14t bloom_filter.py
.PHONY: run
