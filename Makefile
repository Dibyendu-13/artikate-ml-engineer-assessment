.PHONY: all setup build-rag eval-rag train-classifier eval-classifier latency test clean

all: build-rag eval-rag train-classifier eval-classifier latency test

setup:
	python3 -m venv .venv
	. .venv/bin/activate && pip install -U pip && pip install -r requirements.txt

build-rag:
	python -m src.rag_pipeline --build

eval-rag:
	python -m src.rag_pipeline --eval

train-classifier:
	python -m src.classifier.train

eval-classifier:
	python -m src.classifier.evaluate

latency:
	python -m src.classifier.latency_test

test:
	pytest -q

clean:
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.pyc' -delete
	rm -rf .venv
