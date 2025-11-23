.PHONY: setup train-image train-maze analysis-image clean

setup:
	pixi install

train-image:
	pixi run python -m tasks.image_classification.train

train-maze:
	pixi run python -m tasks.mazes.train

analysis-image:
	pixi run python tasks/image_classification/analysis/run_imagenet_analysis.py

clean:
	rm -rf .pixi
