PYTHON ?= python
SIGNSTREAM ?= signstream
CONFIG_DATA ?= configs/data.yaml
CONFIG_MODEL ?= configs/model.yaml
CONFIG_EVAL ?= configs/eval.yaml
CKPT ?= artifacts/checkpoints/pose_transformer_signer_holdout.pt

.PHONY: install data train eval label-budget decode onnx bench test regress all clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

data:
	$(SIGNSTREAM) build-data --config $(CONFIG_DATA)

train:
	$(SIGNSTREAM) train --config $(CONFIG_MODEL) --model pose_transformer --split signer_holdout
	$(SIGNSTREAM) train --config $(CONFIG_MODEL) --model pose_transformer --split random
	$(SIGNSTREAM) train --config $(CONFIG_MODEL) --model bigru --split signer_holdout
	$(SIGNSTREAM) train --config $(CONFIG_MODEL) --model bigru --split random
	$(SIGNSTREAM) train --config $(CONFIG_MODEL) --model linear_probe --split signer_holdout
	$(SIGNSTREAM) train --config $(CONFIG_MODEL) --model linear_probe --split random

eval:
	$(SIGNSTREAM) evaluate --checkpoint $(CKPT) --split signer_holdout --config $(CONFIG_EVAL)
	$(SIGNSTREAM) evaluate --checkpoint artifacts/checkpoints/pose_transformer_random.pt --split random --config $(CONFIG_EVAL)
	$(SIGNSTREAM) evaluate --checkpoint artifacts/checkpoints/bigru_signer_holdout.pt --split signer_holdout --config $(CONFIG_EVAL)
	$(SIGNSTREAM) evaluate --checkpoint artifacts/checkpoints/bigru_random.pt --split random --config $(CONFIG_EVAL)
	$(SIGNSTREAM) evaluate --checkpoint artifacts/checkpoints/linear_probe_signer_holdout.pt --split signer_holdout --config $(CONFIG_EVAL)
	$(SIGNSTREAM) evaluate --checkpoint artifacts/checkpoints/linear_probe_random.pt --split random --config $(CONFIG_EVAL)

label-budget:
	$(SIGNSTREAM) label-budget --model pose_transformer --config $(CONFIG_MODEL)

decode:
	$(SIGNSTREAM) decode --input artifacts/data/continuous/ --threshold 0.6 --checkpoint $(CKPT) --config $(CONFIG_EVAL)

onnx:
	$(SIGNSTREAM) export-onnx --checkpoint $(CKPT)

bench: onnx
	$(SIGNSTREAM) bench --all --checkpoint $(CKPT)

test:
	pytest -q --cov=src/signstream --cov-report=term-missing

regress:
	$(SIGNSTREAM) regress --config configs/thresholds.yaml

all: data train label-budget eval decode onnx bench test regress

clean:
	rm -rf artifacts/data artifacts/checkpoints artifacts/runs artifacts/onnx mlruns .pytest_cache
