.PHONY: test lint serve eval

test:
	pytest -q

lint:
	ruff check . && ruff format .

serve:
	uvicorn scamguard.api:app --reload

eval:
	python -m eval.run_eval
